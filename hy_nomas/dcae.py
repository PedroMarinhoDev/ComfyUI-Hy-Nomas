"""The DCAE-V f32t4c32 encoder: RGB video -> the 32-channel latents the DiT works in (32x in space, 4x in time).

Port of the encoder half of `hymm/models/vae/autoencoder_dcae_v.py` (NVIDIA's DC-AE-V as vendored by Tencent, code
Apache-2.0), with the `dc_ae_v_f32t4c32_config` layout. The DiT's latents are the encoder's output as is
(scaling factor 1, no shift). Hy-Nomas only needs it to start a world from an image (`session.start_world`).

Layout: a 3x3x3 conv to 64 channels + 2x2 pixel unshuffle (256 channels at half size), then stages of residual
blocks (conv, SiLU, conv, RMSNorm) at widths 256/512/512/1024/1024/1024 and depths 5/10/4/4/4/4, each but the last
ending in a downsampling block (conv + pixel unshuffle, plus a channel-averaging shortcut): 2x2 in space four
times, then 4x in time; a last conv to 32 channels.

Every conv is chunk-causal: input whose length is a multiple of the chunk (40 frames, 10 after the temporal
downsample) is cut into chunks, each padded on the left with the previous chunk's last frame (zeros for the first)
and on the right with zeros; any other length is convolved in one piece with zero padding. Video is encoded in
tiles of 40 frames that hand their last frames on to the next (`temporal_tiled_encode`), so frames must come in
multiples of 4 and the result has frames / 4 latent frames.

Weights: the `encoder.*` tensors of the PixelUMM file (`VAE/PixelUMM_FT_FP8/...`, bf16 there and identical to
the ones in `VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt`, which loads too). ~0.9 B parameters, 1.8 GB.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops

CHUNK = 40              # frames per causal chunk and per encode tile
LATENT_CHUNK = 10       # the same chunk after the 4x temporal downsample
WIDTHS = (256, 512, 512, 1024, 1024, 1024)          # stages 1..6 (stage 0 is empty)
DEPTHS = (5, 10, 4, 4, 4, 4)
LATENT_CHANNELS = 32


def pixel_unshuffle_3d(x, spatial, temporal):
    b, c, t, h, w = x.shape
    x = x.reshape(b, c, t // temporal, temporal, h // spatial, spatial, w // spatial, spatial)
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(b, c * temporal * spatial * spatial, t // temporal,
                                                     h // spatial, w // spatial)


class RMSNorm3d(nn.Module):
    """Channel RMSNorm with weight and bias (an `nn.LayerNorm` subclass in the reference, eps 1e-5), in the
    input's dtype."""

    def __init__(self, channels, eps=1e-5, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(channels, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x):
        weight = comfy.model_management.cast_to(self.weight, x.dtype, x.device)
        bias = comfy.model_management.cast_to(self.bias, x.dtype, x.device)
        x = x.permute(0, 2, 3, 4, 1)
        x = x / torch.sqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        return (x * weight + bias).permute(0, 4, 1, 2, 3)


class ChunkConv(nn.Module):
    """`ConvLayer3d` with `causal_chunk_length`: 3x3x3 conv, optional RMSNorm and SiLU.

    `cache` (a dict shared by one encode) carries each layer's last input frame from one tile to the next."""

    def __init__(self, cin, cout, chunk, ops, bias=True, norm=False, act=False, dtype=None, device=None):
        super().__init__()
        self.chunk = chunk
        self.conv = ops.Conv3d(cin, cout, 3, padding=(0, 1, 1), bias=bias, dtype=dtype, device=device)
        self.norm = RMSNorm3d(cout, dtype=dtype, device=device) if norm else None
        self.act = act

    def forward(self, x, cache):
        b, c, t, h, w = x.shape
        if t % self.chunk == 0:
            n, length = t // self.chunk, self.chunk
            previous = cache.get(self)
            if previous is None:
                previous = x.new_zeros(b, c, 1, h, w)
            cache[self] = x[:, :, -1:].clone()
            last = x[:, :, length - 1::length]                              # each chunk's last frame
            left = torch.cat([previous, last[:, :, :-1]], dim=2).unsqueeze(3)
            chunks = torch.cat([left, x.reshape(b, c, n, length, h, w), x.new_zeros(b, c, n, 1, h, w)], dim=3)
            chunks = chunks.transpose(1, 2).reshape(b * n, c, length + 2, h, w)
            del left
            y = self.conv(chunks)
            del chunks
            y = y.view(b, n, -1, length, h, w).transpose(1, 2).reshape(b, -1, t, h, w)
        else:
            y = self.conv(F.pad(x, (0, 0, 0, 0, 1, 1)))
        if self.norm is not None:
            y = self.norm(y)
        if self.act:
            y = F.silu(y)
        return y


class ResBlock(nn.Module):
    """`ResidualBlock3d(ResBlock3d)`: x + RMSNorm(conv(SiLU(conv(x))))."""

    def __init__(self, channels, chunk, ops, dtype=None, device=None):
        super().__init__()
        self.main = nn.Module()
        self.main.conv1 = ChunkConv(channels, channels, chunk, ops, bias=True, act=True, dtype=dtype, device=device)
        self.main.conv2 = ChunkConv(channels, channels, chunk, ops, bias=False, norm=True, dtype=dtype, device=device)

    def forward(self, x, cache):
        return x + self.main.conv2(self.main.conv1(x, cache), cache)


class Unshuffle(nn.Module):
    """`ConvPixelUnshuffleDownSampleLayer3d`: conv to cout / (s*s*t) channels, then pixel unshuffle."""

    def __init__(self, cin, cout, spatial, temporal, ops, dtype=None, device=None):
        super().__init__()
        self.spatial, self.temporal = spatial, temporal
        self.conv = ChunkConv(cin, cout // (spatial * spatial * temporal), CHUNK, ops, dtype=dtype, device=device)

    def forward(self, x, cache):
        return pixel_unshuffle_3d(self.conv(x, cache), self.spatial, self.temporal)


class Downsample(nn.Module):
    """`ResidualBlock3d(Unshuffle, PixelUnshuffleChannelAveraging)`."""

    def __init__(self, cin, cout, spatial, temporal, ops, dtype=None, device=None):
        super().__init__()
        self.main = Unshuffle(cin, cout, spatial, temporal, ops, dtype=dtype, device=device)
        self.cout, self.spatial, self.temporal = cout, spatial, temporal

    def forward(self, x, cache):
        shortcut = pixel_unshuffle_3d(x, self.spatial, self.temporal)
        b, c, t, h, w = shortcut.shape
        shortcut = shortcut.view(b, self.cout, c // self.cout, t, h, w).mean(dim=2)
        return self.main(x, cache) + shortcut


class Stage(nn.Module):
    def __init__(self, ops_list):
        super().__init__()
        self.op_list = nn.ModuleList(ops_list)

    def forward(self, x, cache):
        for op in self.op_list:
            x = op(x, cache)
        return x


class DCAEVEncoder(nn.Module):
    def __init__(self, ops=comfy.ops.manual_cast, dtype=None, device=None):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.project_in = Unshuffle(3, WIDTHS[0], 2, 1, ops, **kw)
        stages = [Stage([])]
        for i, (width, depth) in enumerate(zip(WIDTHS, DEPTHS)):
            last = i == len(WIDTHS) - 1
            ops_list = [ResBlock(width, LATENT_CHUNK if last else CHUNK, ops, **kw) for _ in range(depth)]
            if not last:
                spatial, temporal = (1, 4) if i == len(WIDTHS) - 2 else (2, 1)
                ops_list.append(Downsample(width, WIDTHS[i + 1], spatial, temporal, ops, **kw))
            stages.append(Stage(ops_list))
        self.stages = nn.ModuleList(stages)
        self.project_out = ChunkConv(WIDTHS[-1], LATENT_CHANNELS, LATENT_CHUNK, ops, **kw)

    def forward(self, x, cache):
        x = self.project_in(x, cache)
        for stage in self.stages:
            x = stage(x, cache)
        return self.project_out(x, cache)

    def encode(self, video):
        """video [B, 3, T, H, W] in [-1, 1], T a multiple of 4, H and W of 32 -> latents [B, 32, T / 4, H/32, W/32]."""
        if video.shape[2] % 4 or video.shape[3] % 32 or video.shape[4] % 32:
            raise ValueError(f"DCAE-V needs frames in multiples of 4 and a size in multiples of 32, got "
                             f"{tuple(video.shape[2:])}")
        cache = {}
        tiles = [self(video[:, :, i:i + CHUNK], cache)[:, :, :LATENT_CHUNK] for i in range(0, video.shape[2], CHUNK)]
        return torch.cat(tiles, dim=2)


def load_state_dict(path):
    """`encoder.*` tensors of a PixelUMM safetensors file or of the DCAE-V `.pt` checkpoint."""
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as handle:
            state = {k[len("encoder."):]: handle.get_tensor(k) for k in handle.keys() if k.startswith("encoder.")}
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        checkpoint = checkpoint.get("model_state_dict", checkpoint)
        state = {k[len("encoder."):]: v for k, v in checkpoint.items() if k.startswith("encoder.")}
    if not state:
        raise ValueError(f"{path} has no encoder.* tensors: use the PixelUMM file or the DCAE-V checkpoint")
    logging.info("Hy-Nomas: DCAE-V encoder, %d tensors", len(state))
    return state


def activation_bytes(frames, height, width):
    """Rough peak activation memory of one encode tile: the 256-channel half-size stage dominates (input,
    padded chunk copy, conv output, residual), bf16."""
    frames = min(frames, CHUNK)
    return 5 * 256 * frames * (height // 2) * (width // 2) * 2


def load_encoder(path, disable_dynamic=False):
    """A ModelPatcher around the encoder (bf16)."""
    import comfy.model_patcher
    state = load_state_dict(path)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    encoder = DCAEVEncoder(ops=comfy.ops.manual_cast, dtype=torch.bfloat16, device=offload_device)
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(encoder, load_device=load_device, offload_device=offload_device)
    result = encoder.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, strict=True,
                                     assign=patcher.is_dynamic())
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"DCAE-V encoder mismatch: missing {result.missing_keys[:4]}, "
                         f"unexpected {result.unexpected_keys[:4]}")
    patcher.cached_patcher_init = (load_encoder, (path,))
    return patcher


def encode(patcher, video):
    """IMAGE-style frames [T, H, W, 3] in [0, 1] -> latents [1, 32, T / 4, H / 32, W / 32] (float32, CPU)."""
    frames, height, width = video.shape[0], video.shape[1], video.shape[2]
    comfy.model_management.load_models_gpu([patcher], memory_required=activation_bytes(frames, height, width))
    device = patcher.load_device
    x = (video.to(device).float() * 2 - 1).to(torch.bfloat16).movedim(-1, 0).unsqueeze(0)
    with torch.no_grad(), comfy.model_management.cuda_device_context(device):
        latent = patcher.model.encode(x)
    return latent.float().cpu()
