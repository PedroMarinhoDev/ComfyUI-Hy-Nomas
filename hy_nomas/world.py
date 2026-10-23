"""The generated world so far: every finished segment's latent and its cached keys and values.

Hy-Nomas generates a video one segment at a time. After a segment is sampled, the reference runs one more
forward pass over its clean latent, lightly re-noised (`cache_noise_sigma` 0.206897, timestep 206.897), with the
segment's own text, and keeps every block's keys and values (`causal_ar.py`, the "cache_write" stage). Later
segments attend to those keys and values as their history.

`World` is that state as an immutable value, so it can travel between ComfyUI nodes and be cached like any
output: `extend` returns a new world with one more segment.

What the next segment sees (`History`), following `single_gpu_low_mem_inference.yaml`:

* `past_frames` latent frames of history (18), picked by `temporal_context` sampling over all earlier frames
  (`uniform_history_cache.temporal_context_frame_indices`, gamma 2: denser near the present). 0 keeps everything,
  like the 8-GPU `inference.yaml`.
* the text of every segment that still has a frame in the selection, but only its head: the action list at the
  end of a segment's text (`contained_actions`) is hidden from every other segment
  (`training.cross_segment_text_head_only`). Since a segment never reads its own cache, the hidden tail is
  dropped here instead of being stored and masked.
* keys keep the RoPE positions they were written with; the next segment's video continues the frame count.

Storage: int8 with one fp32 scale per token and head (`int8_kv_cache.py`, `kv_cache_quantization: int8`), or bf16.
The archive lives in system RAM. While a segment samples, its selected history is copied to the GPU once (int8)
and each block dequantizes its own slice; the copy is dropped when the sampling run ends.
"""
import dataclasses
import logging
import math
import weakref

import numpy as np
import torch

CACHE_SIGMA = 0.206897


# ---------------------------------------------------------------------------------------------------------------
# history frame selection (uniform_history_cache.py)


def temporal_context_frame_indices(total_frames, past_frames, gamma=2.0):
    count = min(total_frames, past_frames)
    if count == 0:
        return ()
    if count == 1:
        return (total_frames - 1,)
    if count == total_frames:
        return tuple(range(total_frames))
    i = np.arange(count)
    u = i / (count - 1)
    slack = total_frames - count
    rounded = np.rint(slack * u ** gamma)
    offsets = (0, *(min(slack, max(0, int(value))) for value in rounded[1:-1]), slack)
    distances = tuple(index + offset for index, offset in enumerate(offsets))
    return tuple(total_frames - 1 - distance for distance in reversed(distances))


def uniform_frame_indices(total_frames, past_frames):
    count = min(total_frames, past_frames)
    if count == 0:
        return ()
    if count == 1:
        return (total_frames - 1,)
    return tuple(i * (total_frames - 1) // (count - 1) for i in range(count))


def history_frame_indices(total_frames, past_frames, sampling="temporal_context", gamma=2.0):
    if past_frames <= 0 or total_frames <= past_frames:
        return tuple(range(total_frames))
    if sampling == "temporal_context":
        return temporal_context_frame_indices(total_frames, past_frames, gamma)
    if sampling == "uniform":
        return uniform_frame_indices(total_frames, past_frames)
    if sampling == "recent":
        return tuple(range(total_frames - past_frames, total_frames))
    raise ValueError(f"unknown history sampling {sampling!r}")


# ---------------------------------------------------------------------------------------------------------------
# int8 KV (int8_kv_cache.py: absmax per token and head, half away from zero)


def quantize_kv(x):
    """x [L, H, D] -> (int8 [L, H, D], fp32 scale [L, H, 1])."""
    xf = x.float()
    peak = xf.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(peak > 0, peak / 127.0, torch.ones_like(peak))
    scaled = xf / scale
    rounded = torch.where(scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5))
    return rounded.clamp_(-127, 127).to(torch.int8), scale


def dequantize_kv(q, scale, dtype=torch.bfloat16):
    return (q.float() * scale).to(dtype)


@dataclasses.dataclass(frozen=True)
class SegmentRecord:
    """One finished segment's cache, all layers, on the CPU.

    Per layer: keys/values of its video tokens (frames x tokens_per_frame, frame-major) followed by the visible
    head of its text. `k[i]`/`v[i]` are int8 with `k_scale[i]`/`v_scale[i]`, or bf16 with no scales."""
    index: int
    frame_start: int
    frames: int
    tokens_per_frame: int
    text_tokens: int
    k: tuple
    v: tuple
    k_scale: tuple = None
    v_scale: tuple = None

    @property
    def video_tokens(self):
        return self.frames * self.tokens_per_frame

    def nbytes(self):
        tensors = list(self.k) + list(self.v) + list(self.k_scale or ()) + list(self.v_scale or ())
        return sum(t.numel() * t.element_size() for t in tensors)


class History:
    """The keys and values one segment attends to besides its own, ready per layer on the sampling device."""

    def __init__(self, records, frame_ids, total_frames):
        self.records = records
        self.frame_ids = frame_ids
        self.total_frames = total_frames
        self.layout = []          # (record, selected local frames, text tokens)
        for record in records:
            local = [f - record.frame_start for f in frame_ids if record.frame_start <= f < record.frame_start + record.frames]
            if local:
                self.layout.append((record, local, record.text_tokens))
        tpf = {r.tokens_per_frame for r in records}
        if len(tpf) > 1:
            raise ValueError("the world's segments have different resolutions")
        self.tokens_per_frame = tpf.pop() if tpf else 0
        self.tokens = sum(len(local) * self.tokens_per_frame + text for _, local, text in self.layout)
        self.int8 = bool(records) and records[0].k_scale is not None
        self._device_cache = {}

    def __len__(self):
        return self.tokens

    def _indices(self, record, local):
        tpf = record.tokens_per_frame
        frames = torch.tensor(local, dtype=torch.long)
        idx = (frames[:, None] * tpf + torch.arange(tpf)[None]).flatten()
        if record.text_tokens:
            idx = torch.cat([idx, torch.arange(record.video_tokens, record.video_tokens + record.text_tokens)])
        return idx

    def _gather(self, layer):
        """This layer's working set on the CPU: (k, v, k_scale, v_scale) [Lh, H, D] (+ scales)."""
        parts = {"k": [], "v": [], "k_scale": [], "v_scale": []}
        for record, local, _ in self.layout:
            idx = self._indices(record, local)
            parts["k"].append(record.k[layer].index_select(0, idx))
            parts["v"].append(record.v[layer].index_select(0, idx))
            if self.int8:
                parts["k_scale"].append(record.k_scale[layer].index_select(0, idx))
                parts["v_scale"].append(record.v_scale[layer].index_select(0, idx))
        return {key: torch.cat(value) if value else None for key, value in parts.items()}

    def nbytes_per_layer(self):
        if not self.layout:
            return 0
        record = self.layout[0][0]
        heads, dim = record.k[0].shape[1:]
        per_token = 2 * heads * dim * (1 if self.int8 else 2) + (2 * heads * 4 if self.int8 else 0)
        return self.tokens * per_token

    def release(self):
        self._device_cache.clear()

    def layers(self, device, dtype, run_key=None):
        """A function layer -> (k, v) [1, Lh, H, D] in `dtype` on `device`, or None when there is no history.

        The selected tokens of every layer are staged on `device` once per sampling run (`run_key`, any object
        that lives as long as the run: the sampler's sigmas), int8 when the archive is int8, and each layer is
        dequantized when its block asks for it."""
        if not self.layout:
            return None
        key = (str(device), dtype)
        staged = self._device_cache.get(key)
        if staged is None:
            staged = [None] * len(self.records[0].k)
            for i in range(len(staged)):
                part = self._gather(i)
                staged[i] = {k: (t.to(device, non_blocking=True) if t is not None else None) for k, t in part.items()}
            self._device_cache[key] = staged
            logging.info("Hy-Nomas: history of %d tokens (%d frames) staged on %s, %.2f GiB", self.tokens,
                         len(self.frame_ids), device, self.nbytes_per_layer() * len(staged) / 2**30)
            if run_key is not None:
                weakref.finalize(run_key, self._device_cache.pop, key, None)

        def get(layer):
            part = staged[layer]
            if self.int8:
                k = dequantize_kv(part["k"], part["k_scale"], dtype)
                v = dequantize_kv(part["v"], part["v_scale"], dtype)
            else:
                k, v = part["k"].to(dtype), part["v"].to(dtype)
            return k.unsqueeze(0), v.unsqueeze(0)
        return get


@dataclasses.dataclass(frozen=True, eq=False)
class World:
    """Everything a later segment needs from the earlier ones. Immutable: `extend` makes a new one."""
    records: tuple = ()
    latents: tuple = ()          # each finished segment's clean latent [1, 32, T, h, w], CPU
    past_frames: int = 18
    sampling: str = "temporal_context"
    gamma: float = 2.0
    kv_int8: bool = True
    start_frames: int = 0        # latent frames of a start image held still (segment 0 of `start_world`)

    @property
    def total_frames(self):
        return sum(r.frames for r in self.records)

    @property
    def latent_shape(self):
        return tuple(self.latents[0].shape[-2:]) if self.latents else None

    def history(self):
        cached = getattr(self, "_history", None)
        if cached is None:
            frame_ids = history_frame_indices(self.total_frames, self.past_frames, self.sampling, self.gamma)
            cached = History(self.records, frame_ids, self.total_frames)
            object.__setattr__(self, "_history", cached)
        return cached

    def full_latent(self):
        if not self.latents:
            return None
        return torch.cat(list(self.latents), dim=2)

    def nbytes(self):
        return sum(r.nbytes() for r in self.records)

    def describe(self):
        start = f" (segment 0: a start image, {self.start_frames} still)" if self.start_frames else ""
        return (f"{len(self.records)} segment(s){start}, {self.total_frames} latent frames, "
                f"KV archive {self.nbytes() / 2**30:.2f} GiB ({'int8' if self.kv_int8 else 'bf16'})")

    def with_segment(self, record, latent):
        return dataclasses.replace(self, records=self.records + (record,), latents=self.latents + (latent,))


def make_record(index, frame_start, frames, tokens_per_frame, text_tokens, head_tokens, layer_kv, kv_int8):
    """`layer_kv`: per layer (k, v) [Lv + Lt, H, D] on any device -> a CPU `SegmentRecord` of video + text head."""
    video = frames * tokens_per_frame
    keep = video + head_tokens
    ks, vs, kss, vss = [], [], [], []
    for k, v in layer_kv:
        k, v = k[:keep], v[:keep]
        if kv_int8:
            kq, ksc = quantize_kv(k)
            vq, vsc = quantize_kv(v)
            ks.append(kq.cpu())
            vs.append(vq.cpu())
            kss.append(ksc.cpu())
            vss.append(vsc.cpu())
        else:
            ks.append(k.to(torch.bfloat16).cpu())
            vs.append(v.to(torch.bfloat16).cpu())
    return SegmentRecord(index, frame_start, frames, tokens_per_frame, head_tokens, tuple(ks), tuple(vs),
                         tuple(kss) if kv_int8 else None, tuple(vss) if kv_int8 else None)


def cache_latent(latent, seed, sigma=CACHE_SIGMA):
    """The re-noised latent and timestep the cache pass runs at: (1 - s) x + s e, t = 1000 s (bf16 noise, CPU
    generator, like the reference's rollout generator)."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    eps = torch.randn(latent.shape, generator=generator, dtype=torch.bfloat16)
    noisy = ((1.0 - sigma) * latent.float().cpu() + sigma * eps.float()).to(torch.bfloat16)
    return noisy, torch.full((latent.shape[0],), sigma * 1000.0, dtype=torch.float32)
