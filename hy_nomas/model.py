"""The Hy-Nomas DiT (`HY-Nomas-G/2-32324-MMDiT`), inference only, for one segment at a time.

Port of `hymm/models/dit/nomas_cpt.py` (`CPTHYNomasDiffusionTransformer.forward_with_cache`), with the parameter
names unchanged so Tencent's checkpoints load as they are. 54 dual-stream blocks, hidden 2048, 16 heads of 128,
3-D RoPE [16, 56, 56] at theta 256 on video tokens only, a two-block token refiner on the Qwen2.5-VL features,
and one camera module per block.

What one call computes (the reference's `forward_with_cache` for a seg-AR chunk, batch of identical requests):

* queries: the current segment's video tokens and its valid text tokens (padding never takes part in the
  reference either, so the text arrives trimmed to its valid length);
* keys: the visible history (`world.History`, written by earlier segments' cache passes, RoPE already applied at
  their own frame positions) + the current video + the current text;
* the intra-segment action mask (`training.intra_segment_action_mask`): each action's video window sees the
  head of the text and its own action's text, gap windows the head only, text sees the head plus its own action.
  Computed as one dense attention per group of queries that share a key set (`attention_groups`), which is the
  same function as the reference's masked flex attention.

Camera control (`extrinsics_attn`, `per_chunk_camera_attn` with chunk 1): every video token cross-attends to the
single keyboard token of its own latent frame, so the softmax is over one key and the output is that token's
value. The camera module therefore reduces to `proj_keyboard(v(keyboard))` broadcast over the frame, and its
query/key projections, norms and RoPE do not change the result. They are still loaded, so the file stays whole.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ldm.modules.attention
import comfy.model_management
import comfy.patcher_extension

from . import fused, kernels


# ---------------------------------------------------------------------------------------------------------------
# small pieces


def timestep_embedding(t, dim, max_period=10000):
    """`embed_layers.timestep_embedding`: [cos | sin] in fp32."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
    args = t.flatten()[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb.view(*t.shape, dim)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, operations, dtype=None, device=None, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden, hidden, bias=True, dtype=dtype, device=device),
        )

    def forward(self, t, dtype):
        # the reference casts the sinusoid to the first linear's compute dtype (bf16 for the int8 file)
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size).to(dtype))


class MLP(nn.Module):
    def __init__(self, dim, hidden, act, operations, dtype=None, device=None):
        super().__init__()
        self.fc1 = operations.Linear(dim, hidden, bias=True, dtype=dtype, device=device)
        self.act = act
        self.fc2 = operations.Linear(hidden, dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def fast_int8(self, x):
        return (isinstance(self.act, nn.GELU) and hasattr(self.fc1, "fast_int8") and self.fc1.fast_int8(x)
                and self.fc2.fast_int8(x))

    def forward_int8(self, qa, sa, like):
        """From int8 activations (`kernels.adaln_quant`): fc1 with GELU and the int8 quantization of its output in
        the GEMM epilogue, then fc2. Returns [B, L, dim] bf16 shaped like `like`."""
        with self.fc1.int8_weights(like) as (w1, s1, b1):
            qh, sh = kernels.block_gemm(qa, sa, w1, s1, b1, epilogue="gelu_quant")
        with self.fc2.int8_weights(like) as (w2, s2, b2):
            out = kernels.block_gemm(qh, sh, w2, s2, b2, out_dtype=like.dtype)
        return out.view(*like.shape[:-1], out.shape[-1])


class RMSNorm(nn.Module):
    """`norm_layers.RMSNorm`: normalize in fp32, cast back, then scale (in the input dtype)."""

    def __init__(self, dim, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x):
        out = x.float()
        out = (out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * comfy.model_management.cast_to(self.weight, dtype=out.dtype, device=out.device)


def layer_norm(x, eps=1e-6):
    """`nn.LayerNorm(elementwise_affine=False, eps=1e-6)`. The reference runs under `torch.autocast(bf16)`, which
    computes layer norms in fp32 and keeps the result in fp32 until the next linear casts it."""
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps)


def modulate(x, shift, scale):
    """In fp32 when `x` is a fp32 norm (as under autocast); the caller casts once for the following linear."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def rope_freqs(rope_dims, theta, t, h, w, t_offset=0, device=None):
    """`get_nd_rotary_pos_embed(..., use_real=True)` on the grid [t_offset, t_offset + t) x [0, h) x [0, w).

    Same arithmetic as the reference: an fp32 linspace grid, fp32 frequencies, cos/sin repeat-interleaved."""
    axes = []
    for start, n in ((t_offset, t), (0, h), (0, w)):
        axes.append(torch.linspace(start, start + n, n + 1, dtype=torch.float32, device=device)[:n])
    grid = torch.meshgrid(*axes, indexing="ij")
    cos, sin = [], []
    for dim, pos in zip(rope_dims, grid):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: dim // 2].float() / dim))
        angles = torch.outer(pos.reshape(-1), freqs)
        cos.append(angles.cos().repeat_interleave(2, dim=1))
        sin.append(angles.sin().repeat_interleave(2, dim=1))
    return torch.cat(cos, dim=1), torch.cat(sin, dim=1)


def apply_rope(x, cos, sin):
    """`posemb_layers.apply_rotary_emb` with a (cos, sin) pair: x [B, L, H, D], fp32 math, cast back."""
    xf = x.float()
    real, imag = xf.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-imag, real], dim=-1).flatten(3)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return (xf * cos + rotated * sin).type_as(x)


# ---------------------------------------------------------------------------------------------------------------
# attention


def attention(q, k, v, transformer_options=None):
    """q [B, Lq, H, D], k/v [B, Lk, H, D] -> [B, Lq, H, D]: ComfyUI's attention backend (sage/flash/pytorch)."""
    heads = q.shape[2]
    out = comfy.ldm.modules.attention.optimized_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), heads, skip_reshape=True, skip_output_reshape=True,
        transformer_options=transformer_options or {})
    return out.transpose(1, 2)


class AttentionPlan:
    """The query groups of one segment's attention: which current tokens query which key set.

    `groups` is a list of (query index tensor, text key index tensor) over the concatenated current sequence
    [video (Lv) | text (Lt)]; every group's keys are history + all current video + its text indices. One group
    with every token is the plain bidirectional case. Built once per call from the conditioning's action layout.
    """

    def __init__(self, video_tokens, text_tokens, offsets=None, windows=None, tokens_per_frame=1, device=None):
        self.video_tokens = video_tokens
        self.text_tokens = text_tokens
        lv, lt = video_tokens, text_tokens
        if offsets is None or not windows:
            self.groups = None
            return
        offs = [min(int(o), lt) for o in offsets]
        num_actions = len(offs) - 1
        head = offs[0]
        video_of = {a: [] for a in range(-1, num_actions)}
        for vs, ve, a in windows:
            vs, ve = max(0, min(int(vs) * tokens_per_frame, lv)), max(0, min(int(ve) * tokens_per_frame, lv))
            if ve > vs:
                video_of[a if 0 <= a < num_actions else -1].append((vs, ve))
        groups = []
        for a in range(-1, num_actions):
            spans = list(video_of[a])
            if a == -1:
                text_q = (0, head)
                text_k = [(0, head)]
            else:
                text_q = (offs[a], offs[a + 1])
                text_k = [(0, head), (offs[a], offs[a + 1])]
            if text_q[1] > text_q[0]:
                spans.append((lv + text_q[0], lv + text_q[1]))
            if not spans:
                continue
            q_idx = torch.cat([torch.arange(s, e) for s, e in spans])
            k_idx = torch.cat([torch.arange(s, e) for s, e in text_k if e > s] or [torch.zeros(0, dtype=torch.long)])
            groups.append((q_idx.to(device), k_idx.to(device)))
        covered = sum(g[0].numel() for g in groups)
        if covered != lv + lt:
            raise ValueError(f"action windows cover {covered} of {lv + lt} current tokens")
        self.groups = groups


def segment_attention(q, k, v, history, plan, transformer_options):
    """q/k/v: the current tokens [B, Lv + Lt, H, D]; history: (hk, hv) [1 or B, Lh, H, D] or None."""
    batch = q.shape[0]
    if history is not None:
        hk, hv = history
        if hk.shape[0] != batch:
            hk, hv = hk.expand(batch, -1, -1, -1), hv.expand(batch, -1, -1, -1)
    if plan.groups is None:
        keys = k if history is None else torch.cat([hk, k], dim=1)
        values = v if history is None else torch.cat([hv, v], dim=1)
        return attention(q, keys, values, transformer_options)
    lv = plan.video_tokens
    out = torch.empty_like(q)
    for q_idx, text_idx in plan.groups:
        parts_k = [k[:, :lv]]
        parts_v = [v[:, :lv]]
        if history is not None:
            parts_k.insert(0, hk)
            parts_v.insert(0, hv)
        if text_idx.numel():
            parts_k.append(k[:, lv + text_idx])
            parts_v.append(v[:, lv + text_idx])
        out[:, q_idx] = attention(q[:, q_idx], torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1),
                                  transformer_options)
    return out


# ---------------------------------------------------------------------------------------------------------------
# blocks


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio, operations, dtype=None, device=None):
        super().__init__()
        self.heads = heads
        head_dim = hidden // heads
        mlp_hidden = int(hidden * mlp_ratio)
        kw = dict(dtype=dtype, device=device)
        for side in ("img", "txt"):
            setattr(self, f"{side}_mod", Modulate(hidden, 6, operations, **kw))
            for name in ("q", "k", "v"):
                setattr(self, f"{side}_attn_{name}", operations.Linear(hidden, hidden, bias=True, **kw))
            setattr(self, f"{side}_attn_q_norm", RMSNorm(head_dim, **kw))
            setattr(self, f"{side}_attn_k_norm", RMSNorm(head_dim, **kw))
            setattr(self, f"{side}_attn_proj", operations.Linear(hidden, hidden, bias=True, **kw))
            setattr(self, f"{side}_mlp", MLP(hidden, mlp_hidden, nn.GELU(approximate="tanh"), operations, **kw))

    def _qkv(self, side, x, vec, rope=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = getattr(self, f"{side}_mod")(vec).chunk(6, dim=-1)
        fast = fused.available(x)
        b, n, _ = x.shape
        linears = [getattr(self, f"{side}_attn_{name}") for name in ("q", "k", "v")]
        if fast and all(hasattr(lin, "fast_int8") and lin.fast_int8(x) for lin in linears):
            # one fused LayerNorm/modulate/int8 quantization shared by the three projections
            qa, sa = kernels.adaln_quant(x, shift1, scale1)
            outs = []
            for lin in linears:
                with lin.int8_weights(x) as (w, s, bias):
                    outs.append(kernels.block_gemm(qa, sa, w, s, bias, out_dtype=x.dtype).view(b, n, self.heads, -1))
            q, k, v = outs
        else:
            x_mod = fused.adaln(x, shift1, scale1) if fast else modulate(layer_norm(x), shift1, scale1).to(x.dtype)
            q = linears[0](x_mod).view(b, n, self.heads, -1)
            k = linears[1](x_mod).view(b, n, self.heads, -1)
            v = linears[2](x_mod).view(b, n, self.heads, -1)
        q_norm, k_norm = getattr(self, f"{side}_attn_q_norm"), getattr(self, f"{side}_attn_k_norm")
        if fast:
            q = fused.qk_norm_rope(q, comfy.model_management.cast_to(q_norm.weight, device=q.device), rope)
            k = fused.qk_norm_rope(k, comfy.model_management.cast_to(k_norm.weight, device=k.device), rope)
        else:
            q = q_norm(q).to(v)
            k = k_norm(k).to(v)
            if rope is not None:
                q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        return q, k, v, (gate1, shift2, scale2, gate2)

    def _post(self, side, x, attn, mods):
        gate1, shift2, scale2, gate2 = mods
        proj = getattr(self, f"{side}_attn_proj")
        mlp = getattr(self, f"{side}_mlp")
        if fused.available(x):
            x = fused.gated_residual(x, proj(attn), gate1)
            if mlp.fast_int8(x):
                return fused.gated_residual(x, mlp.forward_int8(*kernels.adaln_quant(x, shift2, scale2), like=x), gate2)
            return fused.gated_residual(x, mlp(fused.adaln(x, shift2, scale2)), gate2)
        x = x + proj(attn) * gate1.unsqueeze(1)
        x = x + mlp(modulate(layer_norm(x), shift2, scale2).to(x.dtype)) * gate2.unsqueeze(1)
        return x

    def forward(self, img, txt, vec, rope, history, plan, transformer_options, capture=None):
        img_q, img_k, img_v, img_mods = self._qkv("img", img, vec, rope)
        txt_q, txt_k, txt_v, txt_mods = self._qkv("txt", txt, vec)
        q = torch.cat([img_q, txt_q], dim=1)
        k = torch.cat([img_k, txt_k], dim=1)
        v = torch.cat([img_v, txt_v], dim=1)
        del img_q, img_k, img_v, txt_q, txt_k, txt_v
        if capture is not None:
            capture(k, v)
        attn = segment_attention(q, k, v, history, plan, transformer_options)
        del q, k, v
        lv = img.shape[1]
        b = img.shape[0]
        img = self._post("img", img, attn[:, :lv].reshape(b, lv, -1), img_mods)
        txt = self._post("txt", txt, attn[:, lv:].reshape(b, attn.shape[1] - lv, -1), txt_mods)
        return img, txt


class Modulate(nn.Module):
    """`ModulateDiT`: linear(silu(vec))."""

    def __init__(self, hidden, factor, operations, dtype=None, device=None):
        super().__init__()
        self.linear = operations.Linear(hidden, factor * hidden, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear(F.silu(x))


class CameraModule(nn.Module):
    """`camera.action_module.ActionModule` (extrinsics_attn), evaluated for one keyboard token per latent frame."""

    def __init__(self, img_hidden, operations, dtype=None, device=None, keyboard_dim=64, hidden=128,
                 keyboard_hidden=1024, heads=16):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.keyboard_hidden = keyboard_hidden
        self.keyboard_embed = nn.Sequential(operations.Linear(keyboard_dim, hidden, bias=True, **kw), nn.SiLU(),
                                            operations.Linear(hidden, hidden, bias=True, **kw))
        self.key_attn_q_norm = RMSNorm(keyboard_hidden // heads, **kw)
        self.key_attn_k_norm = RMSNorm(keyboard_hidden // heads, **kw)
        self.img_attn_q = operations.Linear(img_hidden, keyboard_hidden, bias=False, **kw)
        self.keyboard_attn_kv = operations.Linear(hidden, keyboard_hidden * 2, bias=False, **kw)
        self.proj_keyboard = operations.Linear(keyboard_hidden, img_hidden, bias=False, **kw)

    def frame_output(self, keyboard):
        """keyboard [B, T, 64] -> what each frame's video tokens receive, [B, T, img_hidden]."""
        kv = self.keyboard_attn_kv(self.keyboard_embed(keyboard))
        value = kv[..., self.keyboard_hidden:]
        return self.proj_keyboard(value)


class TokenRefinerBlock(nn.Module):
    def __init__(self, hidden, heads, operations, dtype=None, device=None, mlp_ratio=4.0):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.heads = heads
        self.norm1 = operations.LayerNorm(hidden, elementwise_affine=True, eps=1e-6, **kw)
        self.self_attn_qkv = operations.Linear(hidden, hidden * 3, bias=True, **kw)
        self.self_attn_proj = operations.Linear(hidden, hidden, bias=True, **kw)
        self.norm2 = operations.LayerNorm(hidden, elementwise_affine=True, eps=1e-6, **kw)
        self.mlp = MLP(hidden, int(hidden * mlp_ratio), nn.SiLU(), operations, **kw)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden, 2 * hidden, bias=True, **kw))

    def forward(self, x, c, transformer_options):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        b, n, _ = x.shape
        q, k, v = self.self_attn_qkv(self._norm(self.norm1, x)).view(b, n, 3, self.heads, -1).unbind(2)
        attn = attention(q, k, v, transformer_options).reshape(b, n, -1)
        x = x + self.self_attn_proj(attn) * gate_msa.unsqueeze(1)
        x = x + self.mlp(self._norm(self.norm2, x)) * gate_mlp.unsqueeze(1)
        return x

    @staticmethod
    def _norm(norm, x):
        """An affine LayerNorm as autocast runs it: fp32 input, weights and output, cast for the next linear."""
        weight = comfy.model_management.cast_to(norm.weight, dtype=torch.float32, device=x.device)
        bias = comfy.model_management.cast_to(norm.bias, dtype=torch.float32, device=x.device)
        return F.layer_norm(x.float(), (x.shape[-1],), weight, bias, norm.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden, operations, dtype=None, device=None):
        super().__init__()
        self.linear_1 = operations.Linear(in_dim, hidden, bias=True, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(hidden, hidden, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class TokenRefiner(nn.Module):
    """`SingleTokenRefinerPerBlock` on valid tokens only (the reference's masked attention and masked mean)."""

    def __init__(self, in_dim, hidden, heads, operations, dtype=None, device=None, depth=2):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.input_embedder = operations.Linear(in_dim, hidden, bias=True, **kw)
        self.t_embedder = TimestepEmbedder(hidden, operations, **kw)
        self.c_embedder = TextProjection(in_dim, hidden, operations, **kw)
        self.individual_token_refiner = nn.Module()
        self.individual_token_refiner.blocks = nn.ModuleList(
            [TokenRefinerBlock(hidden, heads, operations, **kw) for _ in range(depth)])

    def forward(self, x, t, dtype, transformer_options):
        timestep_aware = self.t_embedder(t, dtype)
        # the reference's masked mean under autocast: `sum` runs in fp32, so the mean is fp32 until the linear
        context_aware = self.c_embedder((x.float().sum(dim=1) / x.shape[1]).to(x.dtype))
        c = timestep_aware + context_aware
        x = self.input_embedder(x)
        for block in self.individual_token_refiner.blocks:
            x = block(x, c, transformer_options)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, out_channels, operations, dtype=None, device=None):
        super().__init__()
        self.linear = operations.Linear(hidden, out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),
                                              operations.Linear(hidden, 2 * hidden, bias=True, dtype=dtype,
                                                                device=device))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        return self.linear(modulate(layer_norm(x), shift, scale).to(x.dtype))


# ---------------------------------------------------------------------------------------------------------------
# the model


class HyNomasDiT(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, hidden_size=2048, heads_num=16, mlp_width_ratio=4,
                 mm_double_blocks_depth=54, rope_dim_list=(16, 56, 56), rope_theta=256, text_states_dim=3584,
                 camera=True, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.out_channels = out_channels
        self.rope_dim_list = list(rope_dim_list)
        self.rope_theta = rope_theta
        kw = dict(dtype=dtype, device=device)
        self.img_in = nn.Module()
        self.img_in.proj = operations.Conv3d(in_channels, hidden_size, kernel_size=1, stride=1, bias=True, **kw)
        self.txt_in = TokenRefiner(text_states_dim, hidden_size, heads_num, operations, **kw)
        self.time_in = TimestepEmbedder(hidden_size, operations, **kw)
        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(hidden_size, heads_num, mlp_width_ratio, operations, **kw)
             for _ in range(mm_double_blocks_depth)])
        self.final_layer = FinalLayer(hidden_size, out_channels, operations, **kw)
        self.has_camera = camera
        if camera:
            self.action_modules = nn.ModuleList(
                [CameraModule(hidden_size, operations, **kw) for _ in range(mm_double_blocks_depth)])
            self.extrinsics_embedding_camctrl = operations.Linear(64, 64, bias=True, **kw)
            self.c2ws_hidden_states_layer1 = operations.Linear(64, 64, bias=True, **kw)
            self.c2ws_hidden_states_layer2 = operations.Linear(64, 64, bias=True, **kw)

    # -- camera

    def camera_tokens(self, extrinsics, dtype):
        """`_preprocess_camera` for extrinsics_attn: normalized [B, F, 4, 4] -> [B, F'/4, 64] keyboard tokens.

        F is padded at the front with copies of the first matrix to a multiple of 4, then every 4 frames' matrices
        become one 64-d token (one per latent frame)."""
        ext = extrinsics.to(dtype)
        pad = (4 - ext.shape[1] % 4) % 4
        if pad:
            ext = torch.cat([ext[:, :1].repeat(1, pad, 1, 1), ext], dim=1)
        b, f = ext.shape[:2]
        tokens = self.extrinsics_embedding_camctrl(ext.reshape(b, f // 4, 64))
        return tokens + self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(tokens)))

    # -- forward

    def forward(self, x, timestep, context=None, nomas=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward, self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                                     transformer_options)
        ).execute(x, timestep, context, nomas, transformer_options, **kwargs)

    def _forward(self, x, timestep, context=None, nomas=None, transformer_options={}, capture=None, **kwargs):
        """x [B, 32, T, H, W] (the segment's latent), timestep [B] in [0, 1000], context [B or 1, Lt, 3584].

        `nomas` (from the conditioning, `model_base.NomasCond`) carries the action layout, the world history and
        the camera tokens. `capture(layer, k, v)`, when given, receives every block's current keys and values:
        the cache pass of `world.extend` uses it.
        """
        meta = nomas or {}
        b, _, t, h, w = x.shape
        device = x.device
        dtype = self.dtype if self.dtype is not None else x.dtype
        history = meta.get("history")
        frame_offset = history.total_frames if history is not None else 0
        tokens_per_frame = h * w

        rope = rope_freqs(self.rope_dim_list, self.rope_theta, t, h, w, t_offset=frame_offset, device=device)
        img = self.img_in.proj(x.to(dtype)).flatten(2).transpose(1, 2)
        t_in = timestep.to(device)
        vec = self.time_in(t_in, dtype)
        txt = context.to(device=device, dtype=dtype)
        if txt.shape[0] != b:
            txt = txt.expand(b, -1, -1)
        txt = self.txt_in(txt, t_in, dtype, transformer_options)

        plan = AttentionPlan(img.shape[1], txt.shape[1], meta.get("action_offsets"), meta.get("action_windows"),
                             tokens_per_frame, device)

        camera = None
        keyboard = meta.get("camera")
        if keyboard is not None and self.has_camera:
            keyboard = keyboard.to(device=device)
            camera_tokens = self.camera_tokens(keyboard, dtype)
            if camera_tokens.shape[1] != t:
                raise ValueError(f"camera trajectory gives {camera_tokens.shape[1]} latent frames, the segment has {t}")
            if camera_tokens.shape[0] != b:
                camera_tokens = camera_tokens.expand(b, -1, -1)
            camera = camera_tokens

        layers = history.layers(device, dtype, meta.get("run_key")) if history is not None else None
        patches = transformer_options.get("patches", {})
        for index, block in enumerate(self.double_blocks):
            comfy.model_management.throw_exception_if_processing_interrupted()
            hist = layers(index) if layers is not None else None
            cap = (lambda k, v, i=index: capture(i, k, v)) if capture is not None else None
            img, txt = block(img, txt, vec, rope, hist, plan, transformer_options, capture=cap)
            del hist
            if camera is not None:
                frame_out = self.action_modules[index].frame_output(camera)       # [B, T, hidden]
                img = (img.view(b, t, tokens_per_frame, -1) + frame_out.unsqueeze(2)).view(b, t * tokens_per_frame, -1)
            if "double_block" in patches:
                for p in patches["double_block"]:
                    img, txt = p({"img": img, "txt": txt, "index": index})
        img = self.final_layer(img, vec)
        out = img.view(b, t, h, w, self.out_channels).permute(0, 4, 1, 2, 3)
        return out
