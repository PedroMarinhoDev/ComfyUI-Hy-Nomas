"""Fused elementwise kernels for the DiT block, rounding where the reference's autocast run rounds.

The block's arithmetic between the GEMMs (`model.py`) is a dozen small fp32/bf16 kernels per stream; on a 720p
segment they took about as long as the GEMMs. Each function here is one pass, and reproduces the reference's
dtype steps:

* `adaln`: autocast LayerNorm (fp32 out), `* (1 + scale)` with `1 + scale` rounded to bf16, `+ shift`, one bf16
  rounding at the next linear's input.
* `qk_norm_rope`: RMSNorm in fp32, rounded to bf16, times the bf16 weight (rounded), then interleaved RoPE in fp32,
  rounded (`norm_layers.RMSNorm` + `posemb_layers.apply_rotary_emb`).
* `gated_residual`: `x + y * gate`, each product and sum rounded to bf16.

Rounding is done on the bits (round to nearest even) so the compiler cannot fold a bf16 round trip away.
The torch versions in `model.py` stay the reference; `HY_NOMAS_FUSED=0` disables these.
"""
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("HY_NOMAS_FUSED", "1") != "0"


def available(x):
    return ENABLED and x.is_cuda and x.dtype == torch.bfloat16


@triton.jit
def _rbf16(x):
    """fp32 -> the nearest bf16 value (ties to even), still as fp32."""
    bits = x.to(tl.int32, bitcast=True)
    bits = (bits + (((bits >> 16) & 1) + 0x7FFF)) & -65536
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _adaln_kernel(X, SHIFT, SCALE, Y, L, D: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // L
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    n = xc * tl.rsqrt(var + eps)
    scale = tl.load(SCALE + b * D + cols, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(SHIFT + b * D + cols, mask=mask, other=0.0).to(tl.float32)
    y = n * _rbf16(1.0 + scale)
    y = y + shift
    tl.store(Y + row * D + cols, y.to(tl.bfloat16), mask=mask)


def adaln(x, shift, scale, eps=1e-6):
    """x [B, L, D] bf16, shift/scale [B, D] bf16 -> modulate(layer_norm(x), shift, scale) in bf16."""
    b, l, d = x.shape
    x = x.contiguous()
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _adaln_kernel[(b * l,)](x, shift.contiguous(), scale.contiguous(), y, l, d, eps, BLOCK=triton.next_power_of_2(d),
                            num_warps=8)
    return y


@triton.jit
def _qk_norm_rope_kernel(X, W, COS, SIN, Y, H, L, stride_x, eps, ROPE: tl.constexpr, HALF: tl.constexpr):
    row = tl.program_id(0)                      # one (batch, token, head)
    head = row % H
    token = (row // H) % L
    base = (row // H) * stride_x + head * (2 * HALF)
    i = tl.arange(0, HALF)
    xe = tl.load(X + base + 2 * i).to(tl.float32)
    xo = tl.load(X + base + 2 * i + 1).to(tl.float32)
    r = tl.rsqrt((tl.sum(xe * xe, axis=0) + tl.sum(xo * xo, axis=0)) / (2 * HALF) + eps)
    we = tl.load(W + 2 * i).to(tl.float32)
    wo = tl.load(W + 2 * i + 1).to(tl.float32)
    ne = _rbf16(_rbf16(xe * r) * we)
    no = _rbf16(_rbf16(xo * r) * wo)
    if ROPE:
        ce = tl.load(COS + token * (2 * HALF) + 2 * i)
        co = tl.load(COS + token * (2 * HALF) + 2 * i + 1)
        se = tl.load(SIN + token * (2 * HALF) + 2 * i)
        so = tl.load(SIN + token * (2 * HALF) + 2 * i + 1)
        oe = ne * ce - no * se
        oo = no * co + ne * so
    else:
        oe = ne
        oo = no
    out = (row // H) * (H * 2 * HALF) + head * (2 * HALF)
    tl.store(Y + out + 2 * i, oe.to(tl.bfloat16))
    tl.store(Y + out + 2 * i + 1, oo.to(tl.bfloat16))


def qk_norm_rope(x, weight, rope=None, eps=1e-6):
    """x [B, L, H, D] bf16 (rows may be strided, heads contiguous), weight [D] -> RMSNorm(x) * weight, then RoPE
    with (cos, sin) [L, D] fp32 when given. Returns a contiguous [B, L, H, D] bf16 tensor."""
    b, l, h, d = x.shape
    if x.stride(3) != 1 or x.stride(2) != d or x.stride(0) != l * x.stride(1):
        x = x.contiguous()
    y = torch.empty((b, l, h, d), device=x.device, dtype=torch.bfloat16)
    cos, sin = rope if rope is not None else (y, y)
    _qk_norm_rope_kernel[(b * l * h,)](x, weight.to(torch.bfloat16).contiguous(), cos.contiguous(),
                                       sin.contiguous(), y, h, l, x.stride(1), eps, ROPE=rope is not None,
                                       HALF=d // 2, num_warps=1)
    return y


@triton.jit
def _gated_residual_kernel(X, Y, G, OUT, L, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // L
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + b * D + cols, mask=mask, other=0.0).to(tl.float32)
    out = x + _rbf16(y * g)
    tl.store(OUT + row * D + cols, out.to(tl.bfloat16), mask=mask)


def gated_residual(x, y, gate):
    """x + y * gate[:, None] with bf16 rounding of the product and the sum; x, y [B, L, D], gate [B, D]."""
    b, l, d = x.shape
    x = x.contiguous()
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _gated_residual_kernel[(b * l,)](x, y.contiguous(), gate.contiguous(), out, l, d,
                                     BLOCK=triton.next_power_of_2(d), num_warps=8)
    return out


@triton.jit
def _pixelumm_qk_kernel(X, WT, WHW, COS, SIN, Y, H, stride_x, eps, D: tl.constexpr):
    """One (token, head) of PixelUMM's `_prepare_qk`: fp32 RMSNorm of each half with its own weight (T / HW), then
    MRoPE with rotate-half inside the T (D/2), H (D/4) and W (D/4) sections; one bf16 rounding at the end."""
    row = tl.program_id(0)
    head = row % H
    token = row // H
    base = token * stride_x + head * D
    half: tl.constexpr = D // 2
    quarter: tl.constexpr = D // 4
    eighth: tl.constexpr = D // 8
    # T section: dims [0, D/2), rotate-half pairs (i, i + D/4)
    i = tl.arange(0, quarter)
    t0 = tl.load(X + base + i).to(tl.float32)
    t1 = tl.load(X + base + quarter + i).to(tl.float32)
    rt = tl.rsqrt((tl.sum(t0 * t0, axis=0) + tl.sum(t1 * t1, axis=0)) / half + eps)
    t0 = t0 * rt * tl.load(WT + i).to(tl.float32)
    t1 = t1 * rt * tl.load(WT + quarter + i).to(tl.float32)
    # HW half: dims [D/2, D): H section [D/2, 3D/4) and W section [3D/4, D), each rotate-half over D/8
    j = tl.arange(0, eighth)
    h0 = tl.load(X + base + half + j).to(tl.float32)
    h1 = tl.load(X + base + half + eighth + j).to(tl.float32)
    w0 = tl.load(X + base + half + quarter + j).to(tl.float32)
    w1 = tl.load(X + base + half + quarter + eighth + j).to(tl.float32)
    rhw = tl.rsqrt((tl.sum(h0 * h0, axis=0) + tl.sum(h1 * h1, axis=0) + tl.sum(w0 * w0, axis=0)
                    + tl.sum(w1 * w1, axis=0)) / half + eps)
    h0 = h0 * rhw * tl.load(WHW + j).to(tl.float32)
    h1 = h1 * rhw * tl.load(WHW + eighth + j).to(tl.float32)
    w0 = w0 * rhw * tl.load(WHW + quarter + j).to(tl.float32)
    w1 = w1 * rhw * tl.load(WHW + quarter + eighth + j).to(tl.float32)
    cb = COS + token * D
    sb = SIN + token * D
    out = Y + row * D
    ct0, st0 = tl.load(cb + i), tl.load(sb + i)
    ct1, st1 = tl.load(cb + quarter + i), tl.load(sb + quarter + i)
    tl.store(out + i, (t0 * ct0 - t1 * st0).to(tl.bfloat16))
    tl.store(out + quarter + i, (t1 * ct1 + t0 * st1).to(tl.bfloat16))
    ch0, sh0 = tl.load(cb + half + j), tl.load(sb + half + j)
    ch1, sh1 = tl.load(cb + half + eighth + j), tl.load(sb + half + eighth + j)
    tl.store(out + half + j, (h0 * ch0 - h1 * sh0).to(tl.bfloat16))
    tl.store(out + half + eighth + j, (h1 * ch1 + h0 * sh1).to(tl.bfloat16))
    cw0, sw0 = tl.load(cb + half + quarter + j), tl.load(sb + half + quarter + j)
    cw1, sw1 = tl.load(cb + half + quarter + eighth + j), tl.load(sb + half + quarter + eighth + j)
    tl.store(out + half + quarter + j, (w0 * cw0 - w1 * sw0).to(tl.bfloat16))
    tl.store(out + half + quarter + eighth + j, (w1 * cw1 + w0 * sw1).to(tl.bfloat16))


def pixelumm_qk(x, weight_t, weight_hw, cos, sin, eps=1e-6):
    """x [1, N, H, D] (a projection's output, heads contiguous), cos/sin [N, D] fp32 -> bf16 [1, N, H, D]."""
    b, n, h, d = x.shape
    if b != 1 or x.stride(3) != 1 or x.stride(2) != d:
        x = x.contiguous()
    y = torch.empty((b, n, h, d), device=x.device, dtype=torch.bfloat16)
    _pixelumm_qk_kernel[(n * h,)](x, weight_t.contiguous(), weight_hw.contiguous(), cos.contiguous(),
                                  sin.contiguous(), y, h, x.stride(1), eps, D=d, num_warps=1)
    return y
