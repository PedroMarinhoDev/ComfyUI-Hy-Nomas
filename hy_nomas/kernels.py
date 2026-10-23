"""Block-scaled W8A8 linears on the int8 / FP8 tensor cores (Triton).

Both checkpoints keep one weight scale per output row and 128-wide input group (the DiT: `weight_scale`
[N, K/128]; the PixelUMM decoder: one per 128x128 block, expanded over its rows). Here the activations get the
same input grouping, one scale per token and 128 channels, so every 128-deep slice of the dot product is an
exact int8 (or FP8) product rescaled by two scalars:

    y[m, n] = sum_g  a_scale[m, g] * w_scale[n, g] * sum_{k in g} qa[m, k] * qw[n, k]   (+ bias[n])

int8: absmax / 127, round half away from zero (Tencent's rounding), int32 accumulation per group.
fp8:  absmax / 448 (E4M3), fp32 accumulation. On Ada (sm_89) and newer.

The weights are not touched: the DiT's int8 codes and the decoder's FP8 codes are used as stored.
"""
import torch
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _quantize_kernel(X, Q, S, M, K, stride_xm, stride_qm, stride_sm,
                     IS_INT8: tl.constexpr, ROWS: tl.constexpr):
    pid_m = tl.program_id(0)
    g = tl.program_id(1)
    rows = pid_m * ROWS + tl.arange(0, ROWS)
    cols = g * 128 + tl.arange(0, 128)
    mask = (rows[:, None] < M) & (cols[None, :] < K)
    x = tl.load(X + rows[:, None] * stride_xm + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    peak = tl.max(tl.abs(x), axis=1)
    if IS_INT8:
        scale = tl.where(peak > 0, peak / 127.0, 1.0)
        y = x / scale[:, None]
        y = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
        y = tl.minimum(tl.maximum(y, -127.0), 127.0)
        tl.store(Q + rows[:, None] * stride_qm + cols[None, :], y.to(tl.int8), mask=mask)
    else:
        scale = tl.maximum(peak, 1e-12) / 448.0
        y = x / scale[:, None]
        tl.store(Q + rows[:, None] * stride_qm + cols[None, :], y.to(tl.float8e4nv), mask=mask)
    tl.store(S + rows * stride_sm + g, scale, mask=rows < M)


def quantize_activations(x, kind="int8"):
    """x [M, K] (K % 128 == 0) -> codes [M, K] and fp32 scales [M, K / 128]."""
    m, k = x.shape
    dtype = torch.int8 if kind == "int8" else torch.float8_e4m3fn
    q = torch.empty((m, k), device=x.device, dtype=dtype)
    s = torch.empty((m, k // GROUP), device=x.device, dtype=torch.float32)
    rows = 16
    grid = (triton.cdiv(m, rows), k // GROUP)
    _quantize_kernel[grid](x, q, s, m, k, x.stride(0), q.stride(0), s.stride(0), IS_INT8=kind == "int8", ROWS=rows)
    return q, s


def _configs(block_n=None):
    out = []
    for bm, bn, warps, stages in ((128, 128, 8, 3), (128, 64, 4, 4), (64, 128, 4, 4), (128, 256, 8, 2),
                                  (256, 128, 8, 2), (64, 64, 4, 4), (128, 128, 4, 3)):
        if block_n is None or bn == block_n:
            out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "GROUP_M": 8}, num_warps=warps,
                                     num_stages=stages))
    return out


@triton.jit
def _rbf16(x):
    bits = x.to(tl.int32, bitcast=True)
    bits = (bits + (((bits >> 16) & 1) + 0x7FFF)) & -65536
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _gelu_tanh(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * inner) - 1.0))


@triton.jit
def _gemm_body(A, B, C, SA, SB, BIAS, QS, M, N, K, M_BUCKET,
               stride_am, stride_bn, stride_cm, stride_sam, stride_sbn, stride_qsm,
               HAS_BIAS: tl.constexpr, IS_INT8: tl.constexpr, EPILOGUE: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    """EPILOGUE 0: C = A @ B^T (+ bias) in C's dtype. 1: bf16 GELU-tanh of the bf16-rounded result. 2: as 1, then
    quantized to int8 per row and 128 columns (BLOCK_N == 128): C int8, QS [M, N / 128] fp32 scales."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    in_group = GROUP_M * num_n
    group = pid // in_group
    first_m = group * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % in_group) % size_m
    pid_n = (pid % in_group) // size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_ptr = A + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptr = B + offs_n[:, None] * stride_bn + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    groups = K // 128
    for g in range(0, groups):
        a = tl.load(a_ptr, mask=m_mask[:, None])
        b = tl.load(b_ptr, mask=n_mask[:, None])
        if IS_INT8:
            part = tl.dot(a, tl.trans(b), out_dtype=tl.int32).to(tl.float32)
        else:
            part = tl.dot(a, tl.trans(b))
        sa = tl.load(SA + offs_m * stride_sam + g, mask=m_mask, other=0.0)
        sb = tl.load(SB + offs_n * stride_sbn + g, mask=n_mask, other=0.0)
        acc += part * sa[:, None] * sb[None, :]
        a_ptr += 128
        b_ptr += 128
    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]
    c_ptr = C + offs_m[:, None] * stride_cm + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    if EPILOGUE == 0:
        tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=out_mask)
    else:
        h = _rbf16(_gelu_tanh(_rbf16(acc)))
        if EPILOGUE == 1:
            tl.store(c_ptr, h.to(C.dtype.element_ty), mask=out_mask)
        else:
            h = tl.where(out_mask, h, 0.0)
            peak = tl.max(tl.abs(h), axis=1)
            scale = tl.where(peak > 0, peak / 127.0, 1.0)
            y = h / scale[:, None]
            y = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
            y = tl.minimum(tl.maximum(y, -127.0), 127.0)
            tl.store(c_ptr, y.to(tl.int8), mask=out_mask)
            tl.store(QS + offs_m * stride_qsm + pid_n, scale, mask=m_mask)


_KEY = ["M_BUCKET", "N", "K", "IS_INT8", "EPILOGUE"]
_block_gemm = triton.autotune(configs=_configs(), key=_KEY, cache_results=True)(_gemm_body)
_block_gemm_n128 = triton.autotune(configs=_configs(128), key=_KEY, cache_results=True)(_gemm_body)


def _bucket(m):
    return 1 << max(4, (m - 1).bit_length())


EPILOGUES = {None: 0, "gelu": 1, "gelu_quant": 2}


def block_gemm(qa, sa, qw, sw, bias=None, out_dtype=torch.bfloat16, kind="int8", epilogue=None):
    """Block-scaled GEMM on quantized activations. With epilogue "gelu_quant" returns (int8 [M, N], scales
    [M, N / 128]) ready for the next int8 linear; otherwise [M, N] in `out_dtype`."""
    m, k = qa.shape
    n = qw.shape[0]
    mode = EPILOGUES[epilogue]
    if mode == 2:
        out = torch.empty((m, n), device=qa.device, dtype=torch.int8)
        qs = torch.empty((m, n // GROUP), device=qa.device, dtype=torch.float32)
        kernel = _block_gemm_n128
    else:
        out = torch.empty((m, n), device=qa.device, dtype=out_dtype)
        qs = out
        kernel = _block_gemm
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)  # noqa: E731
    kernel[grid](qa, qw, out, sa, sw, bias if bias is not None else out, qs, m, n, k, min(_bucket(m), 65536),
                 qa.stride(0), qw.stride(0), out.stride(0), sa.stride(0), sw.stride(0), qs.stride(0),
                 HAS_BIAS=bias is not None, IS_INT8=kind == "int8", EPILOGUE=mode)
    return (out, qs) if mode == 2 else out


@triton.jit
def _adaln_quant_kernel(X, SHIFT, SCALE, Q, S, L, D: tl.constexpr, eps, GROUPS: tl.constexpr):
    """LayerNorm + modulate (as fused.adaln, rounded to bf16), then int8 per 128 channels."""
    row = tl.program_id(0)
    b = row // L
    g = tl.arange(0, GROUPS)
    c = tl.arange(0, 128)
    cols = g[:, None] * 128 + c[None, :]
    x = tl.load(X + row * D + cols).to(tl.float32)
    mean = tl.sum(tl.sum(x, axis=1), axis=0) / D
    xc = x - mean
    var = tl.sum(tl.sum(xc * xc, axis=1), axis=0) / D
    n = xc * tl.rsqrt(var + eps)
    scale = tl.load(SCALE + b * D + cols).to(tl.float32)
    shift = tl.load(SHIFT + b * D + cols).to(tl.float32)
    y = _rbf16(n * _rbf16(1.0 + scale) + shift)
    peak = tl.max(tl.abs(y), axis=1)
    s = tl.where(peak > 0, peak / 127.0, 1.0)
    q = y / s[:, None]
    q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(Q + row * D + cols, q.to(tl.int8))
    tl.store(S + row * GROUPS + g, s)


def adaln_quant(x, shift, scale, eps=1e-6):
    """x [B, L, D] bf16, shift/scale [B, D] -> int8 codes [B * L, D] and scales [B * L, D / 128] of the modulated
    LayerNorm, for the q/k/v linears."""
    b, l, d = x.shape
    x = x.contiguous()
    q = torch.empty((b * l, d), device=x.device, dtype=torch.int8)
    s = torch.empty((b * l, d // GROUP), device=x.device, dtype=torch.float32)
    _adaln_quant_kernel[(b * l,)](x, shift.contiguous(), scale.contiguous(), q, s, l, d, eps, GROUPS=d // GROUP,
                                  num_warps=8)
    return q, s


def block_linear(x, qw, sw, bias=None, kind="int8"):
    """F.linear with block-scaled 8-bit weights: x [..., K] -> [..., N] in x's dtype."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if shape[-1] % GROUP or x2.shape[0] == 0:
        from .ops import dequantize_g128
        return torch.nn.functional.linear(x, dequantize_g128(qw, sw, x.dtype), bias)
    with torch.cuda.device(x.device):
        qa, sa = quantize_activations(x2, kind)
        out = block_gemm(qa, sa, qw, sw.float(), bias, x.dtype, kind)
    return out.reshape(*shape[:-1], qw.shape[0])


def int8_g128_linear(x, qw, sw, bias=None):
    return block_linear(x, qw, sw, bias, kind="int8")
