"""The PixelUMM decoder: DCAE-V f32t4c32 latents -> RGB video, one distilled denoising step.

Port of `hymm/models/vae/pixelumm_decoder/{model,autoencoder}.py` (Tencent's port of nv-tlabs/PixelUMM, code
Apache-2.0). The released PixelUMM weights and their derivatives are under the NVIDIA One-Way Noncommercial
license: non-commercial research and evaluation only.

The decoder is a 36-layer Qwen3-8B-shaped transformer (hidden 4096, 32 query / 8 KV heads, T/H/W MRoPE, split T/HW
QK-norm) over 4x16x16 RGB tubelets. Each DCAE latent cell conditions 2x2 tubelets (`cond_proj`). It sees pure noise
plus the latents and predicts the clip (x0) in one call; attention is local in space (each 8x8-token tile sees the
3x3 tiles around it) and full in time. A clip is decoded in windows of 10 latent frames with 5 of overlap, blended
with a raised cosine, from per-latent-frame noise seeded by `seed`.

Weights: `VAE/PixelUMM_FT_FP8/diffusion_pytorch_model_fp8.safetensors` (`pixelumm_fp8_bf16_single_v1`): the MLP and
Q/O projections as FP8 E4M3 with one fp32 scale per 128x128 block, everything else bf16. The same file carries the
DCAE-V encoder (`encoder.*`), unused here. A bf16 file (`VAE/PixelUMM_FT`) loads too.
"""
import dataclasses
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.quant_ops import QuantizedTensor

from . import fused

TUBELET = (4, 16, 16)
TUBE_DIM = 4 * 16 * 16 * 3
LATENT_CHANNELS = 32
LATENT_TOKEN_GRID = 2
FP8_FORMAT = "pixelumm_fp8_bf16_single_v1"
FP8_LAYOUT = "NomasFP8Block128Layout"


@dataclasses.dataclass(frozen=True)
class Arch:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta_t: float = 1_000_000.0
    rope_theta_hw: float = 10_000.0

    @property
    def rope_sections(self):
        return self.head_dim // 2, self.head_dim // 4, self.head_dim // 4


# ---------------------------------------------------------------------------------------------------------------
# FP8 block-128 weights as a ComfyUI quantized layout


def dequantize_fp8_block(qdata, scale, dtype=torch.bfloat16):
    n, k = qdata.shape
    w = qdata.float().view(n // 128, 128, k // 128, 128) * scale[:, None, :, None]
    return w.view(n, k).to(dtype)


try:
    from comfy_kitchen.tensor import BaseLayoutParams, QuantizedLayout, register_layout_class

    class NomasFP8Block128Layout(QuantizedLayout):
        MIN_SM_VERSION = None

        @dataclasses.dataclass(frozen=True)
        class Params(BaseLayoutParams):
            pass

        @classmethod
        def quantize(cls, tensor, scale=None, **kwargs):
            n, k = tensor.shape
            blocks = tensor.float().view(n // 128, 128, k // 128, 128)
            s = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4) / 448.0
            q = (blocks / s[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
            return q, cls.Params(scale=s, orig_dtype=tensor.dtype, orig_shape=(n, k))

        @classmethod
        def dequantize(cls, qdata, params):
            return dequantize_fp8_block(qdata, params.scale, params.orig_dtype)

        @classmethod
        def get_plain_tensors(cls, qtensor):
            return qtensor._qdata, qtensor._params.scale

        @classmethod
        def state_dict_tensors(cls, qdata, params):
            return {".fp8_weight": qdata, ".fp8_scale": params.scale}

    register_layout_class(FP8_LAYOUT, NomasFP8Block128Layout)
except ImportError:
    NomasFP8Block128Layout = None


class DecoderOps(comfy.ops.manual_cast):
    """Cast-on-use ops; the Linear also loads `fp8_weight`/`fp8_scale` pairs.

    `matmul`: "dequant" runs bf16 GEMMs on dequantized weights (the decoder's training precision for the
    activations); "fp8" runs FP8 tensor-core GEMMs (`kernels.block_linear`, activations quantized per token and
    128-group)."""
    @classmethod
    def with_matmul(cls, matmul):
        linear = type("Linear", (cls.Linear,), {"matmul": matmul})
        return type(f"DecoderOps_{matmul}", (cls,), {"Linear": linear})

    class Linear(comfy.ops.manual_cast.Linear):
        matmul = "dequant"

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            nn.Module.__init__(self)
            self.in_features, self.out_features = in_features, out_features
            self.weight = None
            self.bias = None
            self._has_bias = bias
            self._dtype = dtype

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                  error_msgs):
            q = state_dict.pop(prefix + "fp8_weight", None)
            s = state_dict.pop(prefix + "fp8_scale", None)
            w = state_dict.pop(prefix + "weight", None)
            b = state_dict.pop(prefix + "bias", None)
            if q is not None:
                params = NomasFP8Block128Layout.Params(scale=s.float(), orig_dtype=self._dtype or torch.bfloat16,
                                                       orig_shape=tuple(q.shape))
                weight = QuantizedTensor(q, FP8_LAYOUT, params)
            elif w is not None:
                weight = w.to(self._dtype) if self._dtype is not None else w
            else:
                missing_keys.append(prefix + "weight")
                return
            self.weight = nn.Parameter(weight, requires_grad=False)
            if b is not None:
                self.bias = nn.Parameter(b.to(self._dtype) if self._dtype is not None else b, requires_grad=False)
            elif self._has_bias:
                missing_keys.append(prefix + "bias")

        def forward_comfy_cast_weights(self, input):
            with comfy.ops.CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                if isinstance(weight, QuantizedTensor):
                    qdata, scale = weight._qdata, weight._params.scale
                    if self.matmul == "fp8" and input.is_cuda:
                        from .kernels import block_linear
                        return block_linear(input, qdata, scale.repeat_interleave(128, dim=0), bias, kind="fp8")
                    weight = dequantize_fp8_block(qdata, scale, input.dtype)
                return F.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            return self.forward_comfy_cast_weights(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------
# the decoder (model.py)


def patchify(video):
    pt, ph, pw = TUBELET
    b, c, t, h, w = video.shape
    x = video.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    return x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(b, -1, TUBE_DIM)


def unpatchify(tokens, grid):
    b = tokens.shape[0]
    nt, nh, nw = grid
    pt, ph, pw = TUBELET
    x = tokens.reshape(b, nt, nh, nw, pt, ph, pw, 3)
    return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, 3, nt * pt, nh * ph, nw * pw)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def mrope_tables(arch, grid, device):
    positions = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
    thetas = (arch.rope_theta_t, arch.rope_theta_hw, arch.rope_theta_hw)
    cos_parts, sin_parts = [], []
    for position, dim, theta in zip(positions, arch.rope_sections, thetas):
        inv_freq = theta ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        angles = position.flatten().float()[:, None] * inv_freq[None]
        angles = torch.cat((angles, angles), dim=-1)
        cos_parts.append(angles.cos())
        sin_parts.append(angles.sin())
    return torch.cat(cos_parts, dim=-1)[None, :, None, :], torch.cat(sin_parts, dim=-1)[None, :, None, :]


def apply_mrope(x, cos, sin, sections):
    rotated = torch.cat([_rotate_half(part) for part in x.split(tuple(sections), dim=-1)], dim=-1)
    return x * cos + rotated * sin


_LAYOUTS = {}
_COMPILED_FLEX = None


def local_attention_layout(grid, tile, device):
    """(perm, inverse, block_mask) of the sliding-tile attention, or None for at most 3x3 tiles (full attention)."""
    from torch.nn.attention.flex_attention import create_block_mask

    frames, height, width = grid
    tiles_h, tiles_w = -(-height // tile), -(-width // tile)
    if tiles_h <= 3 and tiles_w <= 3:
        return None
    key = (grid, tile, str(device))
    if key not in _LAYOUTS:
        t, h, w = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
        tile_h, tile_w = (h // tile).flatten(), (w // tile).flatten()
        order = (((tile_h * tiles_w + tile_w) * frames + t.flatten()) * tile + (h % tile).flatten()) * tile
        perm = torch.argsort(order + (w % tile).flatten())
        tile_h, tile_w = tile_h[perm], tile_w[perm]
        center_h = tile_h.clamp(1, max(tiles_h - 2, 1))
        center_w = tile_w.clamp(1, max(tiles_w - 2, 1))

        def mask_mod(batch, head, q_idx, kv_idx):
            return ((center_h[q_idx] - tile_h[kv_idx]).abs() <= 1) & ((center_w[q_idx] - tile_w[kv_idx]).abs() <= 1)

        count = frames * height * width
        block_mask = create_block_mask(mask_mod, None, None, count, count, device=device,
                                       _compile=torch.device(device).type == "cuda")
        _LAYOUTS[key] = (perm, torch.argsort(perm), block_mask)
    return _LAYOUTS[key]


def _flex_attention(q, k, v, block_mask):
    from torch.nn.attention.flex_attention import flex_attention
    global _COMPILED_FLEX
    if q.device.type != "cuda" or torch.compiler.is_compiling():
        return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
    if _COMPILED_FLEX is None:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX(q, k, v, block_mask=block_mask, enable_gqa=True)


class RMSNorm(nn.Module):
    """Qwen3RMSNorm: normalize in fp32, cast back, then scale."""

    def __init__(self, dim, eps, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        weight = comfy.model_management.cast_to(self.weight, device=x.device)
        return weight * x.to(dtype)


class Attention(nn.Module):
    QK_CHUNK_BYTES = 16 * 2 ** 20

    def __init__(self, arch, ops, dtype=None, device=None):
        super().__init__()
        self.arch = arch
        hidden, heads, kv, dim = arch.hidden_size, arch.num_heads, arch.num_kv_heads, arch.head_dim
        kw = dict(dtype=dtype, device=device)
        self.q_proj = ops.Linear(hidden, heads * dim, bias=False, **kw)
        self.k_proj = ops.Linear(hidden, kv * dim, bias=False, **kw)
        self.v_proj = ops.Linear(hidden, kv * dim, bias=False, **kw)
        self.o_proj = ops.Linear(heads * dim, hidden, bias=False, **kw)
        self.q_norm_t = RMSNorm(dim // 2, arch.rms_norm_eps, **kw)
        self.q_norm_hw = RMSNorm(dim // 2, arch.rms_norm_eps, **kw)
        self.k_norm_t = RMSNorm(dim // 2, arch.rms_norm_eps, **kw)
        self.k_norm_hw = RMSNorm(dim // 2, arch.rms_norm_eps, **kw)

    def _prepare_qk(self, x, cos, sin, norm_t, norm_hw, dtype):
        """fp32 split QK-norm and MRoPE, in token chunks that bound the fp32 temporaries (same arithmetic)."""
        batch, tokens, heads, head_dim = x.shape
        chunk = max(1, self.QK_CHUNK_BYTES // (batch * heads * head_dim * 4))
        result = torch.empty_like(x, dtype=dtype)
        for first in range(0, tokens, chunk):
            last = min(first + chunk, tokens)
            t_part, hw_part = x[:, first:last].float().chunk(2, dim=-1)
            normalized = torch.cat((norm_t(t_part), norm_hw(hw_part)), dim=-1)
            part = apply_mrope(normalized, cos[:, first:last], sin[:, first:last], self.arch.rope_sections)
            result[:, first:last].copy_(part.to(dtype))
        return result

    def forward(self, x, cos, sin, block_mask=None):
        arch = self.arch
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, arch.num_heads, arch.head_dim)
        k = self.k_proj(x).view(b, n, arch.num_kv_heads, arch.head_dim)
        v = self.v_proj(x).view(b, n, arch.num_kv_heads, arch.head_dim)
        if fused.available(v) and b == 1:
            cos2, sin2 = cos[0, :, 0], sin[0, :, 0]
            w = lambda norm: comfy.model_management.cast_to(norm.weight, device=x.device)  # noqa: E731
            q = fused.pixelumm_qk(q, w(self.q_norm_t), w(self.q_norm_hw), cos2, sin2, self.q_norm_t.eps)
            k = fused.pixelumm_qk(k, w(self.k_norm_t), w(self.k_norm_hw), cos2, sin2, self.k_norm_t.eps)
        else:
            q = self._prepare_qk(q, cos, sin, self.q_norm_t, self.q_norm_hw, v.dtype)
            k = self._prepare_qk(k, cos, sin, self.k_norm_t, self.k_norm_hw, v.dtype)
        if block_mask is not None:
            out = _flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask)
        else:
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                 enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, arch.num_heads * arch.head_dim))


class MLP(nn.Module):
    def __init__(self, arch, ops, dtype=None, device=None):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.gate_proj = ops.Linear(arch.hidden_size, arch.intermediate_size, bias=False, **kw)
        self.up_proj = ops.Linear(arch.hidden_size, arch.intermediate_size, bias=False, **kw)
        self.down_proj = ops.Linear(arch.intermediate_size, arch.hidden_size, bias=False, **kw)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, arch, ops, dtype=None, device=None):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.input_norm = RMSNorm(arch.hidden_size, arch.rms_norm_eps, **kw)
        self.attn = Attention(arch, ops, **kw)
        self.post_attn_norm = RMSNorm(arch.hidden_size, arch.rms_norm_eps, **kw)
        self.mlp = MLP(arch, ops, **kw)

    def forward(self, x, cos, sin, block_mask=None):
        x = x + self.attn(self.input_norm(x), cos, sin, block_mask)
        return x + self.mlp(self.post_attn_norm(x))


class PixelHead(nn.Module):
    def __init__(self, hidden, ops, dtype=None, device=None):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=1e-6, dtype=dtype, device=device)
        self.linear = ops.Linear(hidden, TUBE_DIM, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        weight = comfy.model_management.cast_to(self.norm.weight, device=x.device)
        return self.linear(F.rms_norm(x, (x.shape[-1],), weight, self.norm.eps))


class PixelUMMDecoder(nn.Module):
    """`DCAEVPixelUMMDecoder`: forward(latent [B,32,L,h,w], xt [B,3,4L,32h,32w]) -> x0, same shape as xt."""

    def __init__(self, arch=Arch(), local_attention_tile=8, ops=DecoderOps, dtype=torch.bfloat16, device=None):
        super().__init__()
        self.arch = arch
        self.local_attention_tile = int(local_attention_tile)
        self.dtype = dtype
        kw = dict(dtype=dtype, device=device)
        hidden = arch.hidden_size
        self.embed = ops.Linear(TUBE_DIM, hidden, **kw)
        self.cond_proj = ops.Linear(LATENT_CHANNELS, LATENT_TOKEN_GRID ** 2 * hidden, **kw)
        self.layers = nn.ModuleList(Layer(arch, ops, **kw) for _ in range(arch.num_layers))
        self.final_norm = RMSNorm(hidden, arch.rms_norm_eps, **kw)
        self.head = PixelHead(hidden, ops, **kw)

    def _condition_tokens(self, latent):
        b, _, frames, height, width = latent.shape
        g = LATENT_TOKEN_GRID
        x = self.cond_proj(latent.permute(0, 2, 3, 4, 1))
        x = x.view(b, frames, height, width, g, g, self.arch.hidden_size)
        return x.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, -1, self.arch.hidden_size)

    def forward(self, latent, xt):
        dtype = self.dtype
        _, _, frames, height, width = latent.shape
        grid = (frames, height * LATENT_TOKEN_GRID, width * LATENT_TOKEN_GRID)
        tokens = self.embed(patchify(xt.to(dtype))) + self._condition_tokens(latent.to(dtype))
        cos, sin = mrope_tables(self.arch, grid, latent.device)
        layout = local_attention_layout(grid, self.local_attention_tile, latent.device) if self.local_attention_tile else None
        block_mask = None
        if layout is not None:
            perm, inverse, block_mask = layout
            tokens, cos, sin = tokens[:, perm], cos[:, perm], sin[:, perm]
        for layer in self.layers:
            comfy.model_management.throw_exception_if_processing_interrupted()
            tokens = layer(tokens, cos, sin, block_mask)
        tokens = self.final_norm(tokens)
        if layout is not None:
            tokens = tokens[:, inverse]
        return unpatchify(self.head(tokens), grid)


# ---------------------------------------------------------------------------------------------------------------
# clip decode (autoencoder.py, num_steps 1, no spatial tiles, window parallelism off)


def window_starts(length, window, stride):
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def make_noise(latent, start, end, seed, device):
    """fp32 noise for latent frames [start, end): one generator per frame, seeded `seed * 1_000_003 + index`."""
    b, _, _, h, w = latent.shape
    frames = []
    for index in range(start, end):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed * 1_000_003 + index)
        frames.append(torch.randn((b, 3, 4, h * 32, w * 32), generator=generator, device=device,
                                  dtype=torch.float32))
    return torch.cat(frames, dim=2)


class TemporalBlender:
    """`_TemporalBlender`: windows cosine-blended into one clip buffer as they arrive."""

    def __init__(self, total_frames, output_device):
        self.total = int(total_frames)
        self.output_device = output_device
        self.out = None
        self.end = 0

    def add(self, part, start):
        length = part.shape[2]
        if self.out is None:
            self.out = part.new_empty((*part.shape[:2], self.total, *part.shape[3:]), device=self.output_device)
            self.out[:, :, :length] = part.to(self.output_device)
            self.end = length
            return
        overlap = self.end - start
        position = torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32)
        weight = (0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))).view(1, 1, -1, 1, 1).to(part.dtype)
        previous = self.out[:, :, start:self.end].to(part.device)
        self.out[:, :, start:self.end] = (previous * (1.0 - weight) + part[:, :, :overlap] * weight).to(self.output_device)
        self.out[:, :, self.end:start + length] = part[:, :, overlap:].to(self.output_device)
        self.end = start + length

    def result(self):
        return self.out


def decode_clip(decoder, latent, seed=42, window=10, overlap=5, device=None, output_device="cpu", progress=None,
                index_offset=0):
    """latent [B, 32, L, h, w] -> video [B, 3, 4L, 32h, 32w] in [-1, 1] (decoder dtype) on `output_device`.

    `index_offset`: the absolute latent index of `latent`'s first frame, so a slice of a longer timeline gets the
    same per-frame noise as the whole timeline would."""
    device = device or latent.device
    frames = int(latent.shape[2])
    starts = window_starts(frames, window, window - overlap)
    t_len = min(window, frames)
    blender = TemporalBlender((starts[-1] + t_len) * 4, output_device)
    latent = latent.to(device)
    if decoder.local_attention_tile:
        local_attention_layout((t_len, latent.shape[3] * 2, latent.shape[4] * 2), decoder.local_attention_tile, device)
    for i, t0 in enumerate(starts):
        real = latent[:, :, t0:t0 + t_len].contiguous()
        xt = make_noise(latent, index_offset + t0, index_offset + t0 + t_len, seed, device)
        with torch.no_grad():
            x0 = decoder(real, xt)
        blender.add(x0, t0 * 4)
        if progress is not None:
            progress(i + 1, len(starts))
    return blender.result()


def load_state_dict(path):
    """`decoder.*` tensors of a PixelUMM checkpoint (FP8 single file or bf16), without the DCAE encoder."""
    from safetensors import safe_open
    state = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            if key.startswith("decoder."):
                state[key[len("decoder."):]] = handle.get_tensor(key)
    if not state:
        raise ValueError(f"{path} has no decoder.* tensors: not a PixelUMM checkpoint")
    fp8 = metadata.get("format") == FP8_FORMAT or any(k.endswith(".fp8_weight") for k in state)
    logging.info("Hy-Nomas: PixelUMM decoder %s (%d tensors)", "FP8 block-128" if fp8 else "bf16", len(state))
    return state, fp8


def load_decoder(path, matmul="dequant", disable_dynamic=False):
    """A ModelPatcher around the decoder, so ComfyUI loads and offloads it like any model."""
    import comfy.model_patcher
    state, fp8 = load_state_dict(path)
    ops = DecoderOps.with_matmul(matmul)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    decoder = PixelUMMDecoder(ops=ops, device=offload_device)
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(decoder, load_device=load_device, offload_device=offload_device)
    result = decoder.load_state_dict(state, strict=False, assign=patcher.is_dynamic())
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"PixelUMM checkpoint mismatch: missing {result.missing_keys[:4]}, "
                         f"unexpected {result.unexpected_keys[:4]}")
    patcher.cached_patcher_init = (load_decoder, (path, matmul))
    return patcher


def decode(patcher, latent, seed=42, pixel_frames=None, progress=None, overlap=5):
    """LATENT samples [B, 32, L, h, w] -> IMAGE frames [B * frames, 32h, 32w, 3] in [0, 1] (CPU float32)."""
    comfy.model_management.load_models_gpu([patcher])
    device = patcher.load_device
    decoder = patcher.model
    with comfy.model_management.cuda_device_context(device):
        video = decode_clip(decoder, latent.to(torch.bfloat16), seed=seed, overlap=overlap, device=device,
                            output_device="cpu", progress=progress)
    if pixel_frames is not None:
        video = video[:, :, :pixel_frames]
    video = (video.float() * 0.5 + 0.5).clamp_(0, 1)
    return video.permute(0, 2, 3, 4, 1).reshape(-1, video.shape[3], video.shape[4], 3)


def decode_new_frames(patcher, timeline, start, end, seed=42, window=10, overlap=5):
    """Streaming decode for drive mode: video frames of latents [start, end) of a growing `timeline`
    [1, 32, L, h, w], as uint8 [frames, H, W, 3] (CPU).

    A turn shorter than a window is decoded by one full window ending at `end`, so its frames are decoded with
    earlier latents as context (as the reference's streaming session does); a longer turn by the usual
    overlapping windows from `start - overlap`. The noise follows the absolute latent index, so the frames match
    what a full decode of the timeline would give for that window."""
    if end - start <= window:
        first = max(0, end - window)
    else:
        first = max(0, start - overlap)
    comfy.model_management.load_models_gpu([patcher])
    device = patcher.load_device
    with comfy.model_management.cuda_device_context(device):
        video = decode_clip(patcher.model, timeline[:, :, first:end].to(torch.bfloat16), seed=seed, window=window,
                            overlap=overlap, device=device, output_device=device, index_offset=first)
    video = video[:, :, (start - first) * 4:]
    video = ((video.float() * 0.5 + 0.5).clamp_(0, 1) * 255).round_().to(torch.uint8)
    return video[0].permute(1, 2, 3, 0).cpu()
