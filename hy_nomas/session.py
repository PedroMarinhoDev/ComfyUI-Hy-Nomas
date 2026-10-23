"""The steps between sampler runs: conditioning for a segment, and the cache pass that adds it to the world."""
import dataclasses
import logging

import torch

import comfy.model_management
import comfy.utils

from . import dcae
from . import prompt as P
from . import text as T
from . import world as W


class Script:
    """A parsed world JSON: per-segment prompt strings and frame boundaries (24 fps, 4 frames per latent)."""

    def __init__(self, source, fps=P.FPS):
        self.source = source
        self.world = P.parse_world(source)
        self.prompts = P.segment_prompts(self.world)
        self.fps = fps
        pixel = P.pixel_boundaries(self.world, fps)
        if pixel is None:
            raise ValueError("every segment needs start_time and end_time")
        if pixel[0] != 0:
            pixel = [b - pixel[0] for b in pixel]
        self.pixel_boundaries = pixel
        self.latent_boundaries = P.latent_boundaries(pixel)
        self.num_frames = pixel[-1]

    def __len__(self):
        return len(self.prompts)

    def segment_frames(self, index):
        return self.latent_boundaries[index + 1] - self.latent_boundaries[index]

    def describe(self):
        segs = ", ".join(f"{self.segment_frames(i)}" for i in range(len(self)))
        return (f"{len(self)} segment(s), {self.num_frames} frames at {self.fps} fps "
                f"({self.latent_boundaries[-1]} latent frames: {segs})")


def check_world(world, index, width, height):
    if world is None:
        if index != 0:
            raise ValueError(f"segment {index} needs the world holding segments 0..{index - 1}")
        return
    if len(world.records) != index:
        raise ValueError(f"segment {index} needs a world of {index} segment(s); this one has {len(world.records)}")
    if world.latent_shape is not None and world.latent_shape != (height // 32, width // 32):
        raise ValueError(f"the world was generated at latent {world.latent_shape}, this segment asks for "
                         f"{(height // 32, width // 32)}")


def segment_conditioning(clip, script, index, width, height, world=None, camera=None, text_override=None):
    """CONDITIONING and an empty LATENT for segment `index` of `script`."""
    if not 0 <= index < len(script):
        raise ValueError(f"segment_index {index} is outside the script's {len(script)} segment(s)")
    check_world(world, index, width, height)
    frames = script.segment_frames(index)
    prompt = text_override if text_override else script.prompts[index]
    states, layout = T.encode_segment(clip, prompt, frames)
    layout.update(segment_index=index, frames=frames, world=world)
    if camera is not None:
        layout["camera"] = camera.segment(script, index)
    latent = torch.zeros([1, 32, frames, height // 32, width // 32])
    logging.info("Hy-Nomas: segment %d: %d latent frames, %d text tokens (head %d), actions %s, history %s",
                 index, frames, layout["text_tokens"], T.head_tokens(layout),
                 layout["action_windows"] or "-", world.describe() if world is not None else "none")
    return [[states.to(torch.bfloat16), {"nomas": layout}]], {"samples": latent}


NEGATIVE_PROMPT = ("low quality, worst quality, low resolution, blurry, out of focus, jpeg artifacts, noise, grainy, "
                   "distorted, deformed, bad anatomy, bad proportions, extra limbs, missing limbs, duplicate, "
                   "watermark, text, logo, signature")


def negative_conditioning(clip, positive, text=NEGATIVE_PROMPT):
    """The reference's CFG negative (`inference.neg_prompt`, used only when cfg > 1): the same template around the
    negative text, full attention (no action layout), the positive's history and camera
    (`ar_uncond_with_camera`)."""
    tok = T.prompt_tokenizer(clip)
    states = T.encode_ids(clip, tok.ids(text))
    layout = dict(positive[0][1]["nomas"], text=text, text_tokens=states.shape[1], action_start=-1,
                  action_offsets=None, action_windows=None)
    return [[states.to(torch.bfloat16), {"nomas": layout}]]


class _RecordBuilder:
    """Collects one cache pass's per-layer keys/values straight into CPU storage (int8 or bf16)."""

    def __init__(self, keep_tokens, kv_int8):
        self.keep = keep_tokens
        self.int8 = kv_int8
        self.k, self.v, self.ks, self.vs = {}, {}, {}, {}

    def __call__(self, layer, k, v):
        k, v = k[0, :self.keep], v[0, :self.keep]
        if self.int8:
            kq, ksc = W.quantize_kv(k)
            vq, vsc = W.quantize_kv(v)
            self.k[layer], self.v[layer] = kq.cpu(), vq.cpu()
            self.ks[layer], self.vs[layer] = ksc.cpu(), vsc.cpu()
        else:
            self.k[layer], self.v[layer] = k.to(torch.bfloat16).cpu(), v.to(torch.bfloat16).cpu()

    def record(self, index, frame_start, frames, tokens_per_frame, head):
        layers = sorted(self.k)
        return W.SegmentRecord(index, frame_start, frames, tokens_per_frame, head,
                               tuple(self.k[i] for i in layers), tuple(self.v[i] for i in layers),
                               tuple(self.ks[i] for i in layers) if self.int8 else None,
                               tuple(self.vs[i] for i in layers) if self.int8 else None)


def extend_world(model, conditioning, samples, seed=0, past_frames=18, kv_int8=True, sampling="temporal_context"):
    """Run the cache pass for one sampled segment and return the world with it appended."""
    states, meta = conditioning[0][0], conditioning[0][1]
    layout = meta.get("nomas")
    if layout is None:
        raise ValueError("this conditioning does not come from a Hy-Nomas Segment node")
    world = layout.get("world")
    if world is None:
        world = W.World(past_frames=past_frames, sampling=sampling, kv_int8=kv_int8)
    latent = samples["samples"]
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"expected one segment latent [1, 32, T, h, w], got {tuple(latent.shape)}")
    if latent.shape[2] != layout["frames"]:
        raise ValueError(f"the latent has {latent.shape[2]} frames, the segment {layout['frames']}")

    comfy.model_management.load_models_gpu([model])
    device = model.load_device
    dm = model.model.diffusion_model
    noisy, t = W.cache_latent(latent, seed)
    tokens_per_frame = latent.shape[3] * latent.shape[4]
    head = T.head_tokens(layout)
    builder = _RecordBuilder(latent.shape[2] * tokens_per_frame + head, world.kv_int8)
    history = world.history() if world.records else None
    nomas = {"action_offsets": layout.get("action_offsets"), "action_windows": layout.get("action_windows"),
             "camera": layout.get("camera"), "history": history}
    with torch.no_grad(), comfy.model_management.cuda_device_context(device):
        dm._forward(noisy.to(device), t.to(device), states.to(device), nomas=nomas,
                    transformer_options=model.model_options.get("transformer_options", {}), capture=builder)
    if history is not None:
        history.release()
    record = builder.record(len(world.records), world.total_frames, latent.shape[2], tokens_per_frame, head)
    new = world.with_segment(record, latent.detach().to("cpu", torch.float32))
    logging.info("Hy-Nomas: world now %s", new.describe())
    return new


def still_clip(image, width, height, frames):
    """The first IMAGE [B, H, W, 3] scaled to cover width x height, center-cropped, repeated -> [frames, H, W, 3]."""
    picture = comfy.utils.common_upscale(image[:1].movedim(-1, 1), width, height, "lanczos", "center")
    return picture.movedim(1, -1).clamp(0, 1).repeat(frames, 1, 1, 1)


def image_prompt(item):
    """Segment 0's prompt without its actions: the image's segment shows the world, and nothing happens in it."""
    first = dict(item["segments"][0], contained_actions=[])
    return P._build_whole_item_prompts(item, [first])[0]


def start_world(model, clip, encoder, item, image, width, height, still_frames=3, seed=0, past_frames=18,
                kv_int8=True):
    """A world whose segment 0 is `image`: the picture held still for `still_frames` latent frames (4 video frames
    each), encoded with DCAE-V and added by the cache pass under the text of the world's (`item`, a parsed world
    JSON) first segment (`image_prompt`). The model has no image input; this puts the image where it looks: in the
    memory of what came before. Generation continues with segment 1, which sees the image as its history."""
    if width % 32 or height % 32:
        raise ValueError(f"width and height must be multiples of 32, got {width}x{height}")
    latent = dcae.encode(encoder, still_clip(image, width, height, 4 * still_frames))
    states, layout = T.encode_segment(clip, image_prompt(item), still_frames)
    layout.update(segment_index=0, frames=still_frames, world=None)
    positive = [[states.to(torch.bfloat16), {"nomas": layout}]]
    world = extend_world(model, positive, {"samples": latent}, seed=seed, past_frames=past_frames, kv_int8=kv_int8)
    return dataclasses.replace(world, start_frames=still_frames)


START_IMAGE_OPTIONS = ["one frame", "whole hold", "none"]


def trim_start(images, world, keep="one frame"):
    """Decoded frames of a world started from an image: keep its last held frame (the clip opens on the image and
    moves at once), all of the hold, or none of it."""
    hold = 4 * world.start_frames if world is not None else 0
    if not hold or keep == "whole hold":
        return images
    return images[hold - 1:] if keep == "one frame" else images[hold:]
