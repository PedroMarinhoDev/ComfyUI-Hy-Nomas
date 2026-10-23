"""Hy-Nomas nodes: a world generated one segment at a time with the stock samplers.

    Load DiT ─────────────────────────────┐
    Load CLIP (qwen_image) ─┐             │
    World Prompt ─── Segment 0 ─(cond, latent)─ KSampler ─ Extend World ─ Segment 1 ─ ... ─ Decode World
                                                                  (world)                  (frames, fps)

Sampling contract (Tencent's `inference.yaml`): 8 steps, sampler `euler`, scheduler `simple`, cfg 1. The model
carries the shift (5). Each segment's history (earlier segments' cached keys/values) rides in its conditioning.
"""
import logging

import torch
from typing_extensions import override

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, IO

from .hy_nomas import camera as cam
from .hy_nomas import drive
from .hy_nomas import dcae, loader, pixelumm, session
from .hy_nomas import world as W

SCRIPT = IO.Custom("HYNOMAS_SCRIPT")
WORLD = IO.Custom("HYNOMAS_WORLD")
DECODER = IO.Custom("HYNOMAS_DECODER")
ENCODER = IO.Custom("HYNOMAS_ENCODER")
CAMERA = IO.Custom("HYNOMAS_CAMERA")

CATEGORY = "Hy-Nomas"
MATMUL_OPTIONS = ["int8", "exact"]
DEFAULT_PROMPT = """{
  "perspective_view": "third-person observer",
  "Style": "CG",
  "subject_list": [
    {"subject_id": "<|Subject1|>", "display_name": "small orange fox",
     "subject_attribute": "A small fox with orange fur, a white chest and a fluffy white-tipped tail, standing on a stone path in side view. The camera films the small orange fox from an external position, and the small orange fox is visible in the video. The framing of the video from the camera is independent of the small orange fox, and does not follow the movement of the small orange fox."}
  ],
  "segments": [
    {
      "start_time": "00:00.000", "end_time": "00:05.000",
      "visual_description_long": "A flagstone path crosses a luminous forest of tall dark trunks under a misty canopy. Lush green grass and low bushes line both sides of the path, with mossy gray rocks scattered along its edge. Small blue fireflies drift and glow between the trees in the soft, hazy light, and the background fades into pale fog.",
      "description_change": null,
      "contained_actions": [
        {"start_time": "00:00.000", "end_time": "00:05.000", "action_description": "<Action1><|Subject1|> walks slowly to the left along the stone path</Action1>."}
      ]
    },
    {
      "start_time": "00:05.000", "end_time": "00:10.000",
      "visual_description_long": null,
      "description_change": "Compared to the previous segment, clusters of glowing blue flowers open in the grass beside the path, while the trees, fog and fireflies remain unchanged.",
      "contained_actions": [
        {"start_time": "00:05.000", "end_time": "00:07.500", "action_description": "<Action1><|Subject1|> stops beside the flowers and turns around to face the camera</Action1>."},
        {"start_time": "00:07.500", "end_time": "00:10.000", "action_description": "<Action2><|Subject1|> sniffs the glowing flowers and looks around</Action2>."}
      ]
    }
  ]
}"""

SIZE_TOOLTIP = ("Frame size, a multiple of 32. Tencent's defaults: 1280x704 (their 720p; the latent grid is 40x22) "
                "or 832x480. Every segment of a world must use the same size.")


def _files(folder, needle=""):
    files = folder_paths.get_filename_list(folder)
    picked = [f for f in files if needle in f.lower()]
    return picked or files


def _encoder_files():
    files = folder_paths.get_filename_list("vae")
    picked = [f for f in files if any(n in f.lower() for n in ("pixelumm", "dcae", "dc-ae"))]
    return picked or files


class HyNomasLoadDiT(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasLoadDiT",
            display_name="Hy-Nomas Load DiT",
            category=CATEGORY,
            description="Loads the Hy-Nomas world-model DiT (Tencent's INT8 distilled file, or a bf16/fp32 one) "
                        "from models/diffusion_models.",
            inputs=[
                IO.Combo.Input("dit_name", options=_files("diffusion_models", "nomas") or [""],
                               tooltip="e.g. diffusion_pytorch_model_ema_0_995_int8.safetensors (DiT/distilled_int8)"),
                IO.Combo.Input("matmul", options=MATMUL_OPTIONS, default="int8",
                               tooltip="int8: the int8 tensor cores (W8A8, activations quantized per token and 128 "
                                       "channels; several times faster on RTX 30/40). exact: bf16 matmuls on the "
                                       "dequantized weights, Tencent's arithmetic bit for bit."),
                IO.Boolean.Input("quantize_float", default=True, advanced=True, optional=True,
                                 tooltip="For a bf16/fp32 file: quantize its linears to Tencent's int8 format at "
                                         "load (9 GB instead of 17 GB)."),
            ],
            outputs=[IO.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, dit_name, matmul="int8", quantize_float=True) -> IO.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", dit_name)
        return IO.NodeOutput(loader.load_dit(path, matmul=matmul, quantize=quantize_float))


class HyNomasWorldPrompt(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasWorldPrompt",
            display_name="Hy-Nomas World Prompt",
            category=CATEGORY,
            description="The world as Tencent's structured JSON: Style, perspective_view, subject_list and timed "
                        "segments (visual_description_long for the first, description_change after, optional "
                        "contained_actions). Shaped into one prompt per segment exactly as the reference does.",
            inputs=[IO.String.Input("world_json", multiline=True, default=DEFAULT_PROMPT)],
            outputs=[SCRIPT.Output(display_name="script"), IO.Int.Output(display_name="segments"),
                     IO.Int.Output(display_name="frames"), IO.String.Output(display_name="summary")],
        )

    @classmethod
    def execute(cls, world_json) -> IO.NodeOutput:
        script = session.Script(world_json)
        return IO.NodeOutput(script, len(script), script.num_frames, script.describe())


class HyNomasSegment(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasSegment",
            display_name="Hy-Nomas Segment",
            category=CATEGORY,
            description="Conditioning and an empty latent for one segment: its text (Qwen2.5-VL, layer -3), its "
                        "action masks, and the world so far as history. Sample it with KSampler (euler, simple, "
                        "8 steps, cfg 1), then add it to the world with Extend World.",
            inputs=[
                IO.Clip.Input("clip", tooltip="Qwen2.5-VL-7B in Load CLIP with type qwen_image"),
                SCRIPT.Input("script"),
                IO.Int.Input("segment_index", default=0, min=0, max=1000),
                IO.Int.Input("width", default=832, min=32, max=4096, step=32, tooltip=SIZE_TOOLTIP),
                IO.Int.Input("height", default=480, min=32, max=4096, step=32, tooltip=SIZE_TOOLTIP),
                WORLD.Input("world", optional=True, tooltip="The world holding segments 0..index-1; none for the "
                                                            "first segment."),
                CAMERA.Input("camera", optional=True),
                IO.String.Input("override_text", optional=True, force_input=True, multiline=True,
                                tooltip="A segment prompt string to use instead of the script's."),
            ],
            outputs=[IO.Conditioning.Output(display_name="positive"), IO.Conditioning.Output(display_name="negative"),
                     IO.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, clip, script, segment_index, width, height, world=None, camera=None,
                override_text=None) -> IO.NodeOutput:
        positive, latent = session.segment_conditioning(clip, script, segment_index, width, height, world=world,
                                                        camera=camera, text_override=override_text)
        negative = session.negative_conditioning(clip, positive)
        return IO.NodeOutput(positive, negative, latent)


class HyNomasExtendWorld(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasExtendWorld",
            display_name="Hy-Nomas Extend World",
            category=CATEGORY,
            description="Adds a sampled segment to the world: one forward pass over its latent (lightly re-noised, "
                        "sigma 0.207, as the reference's cache write) keeps every block's keys and values, which "
                        "later segments attend to.",
            inputs=[
                IO.Model.Input("model"),
                IO.Conditioning.Input("positive", tooltip="The Segment node's positive output for this segment"),
                IO.Latent.Input("samples", tooltip="The sampled segment"),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=False,
                             tooltip="Seed of the cache pass's re-noising"),
                IO.Int.Input("history_frames", default=18, min=0, max=1024, advanced=True, optional=True,
                             tooltip="Latent frames of history a segment attends to (temporal_context sampling, "
                                     "denser near the present). 18 is the single-GPU preset; 0 keeps all of it. "
                                     "Read when the world is created (segment 0)."),
                IO.Boolean.Input("int8_kv", default=True, advanced=True, optional=True,
                                 tooltip="Store the history keys/values as int8 (the single-GPU preset) rather than "
                                         "bf16. Read when the world is created."),
            ],
            outputs=[WORLD.Output(display_name="world"), IO.String.Output(display_name="summary")],
        )

    @classmethod
    def execute(cls, model, positive, samples, seed, history_frames=18, int8_kv=True) -> IO.NodeOutput:
        world = session.extend_world(model, positive, samples, seed=seed, past_frames=history_frames,
                                     kv_int8=int8_kv)
        return IO.NodeOutput(world, world.describe())


class HyNomasLoadEncoder(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasLoadEncoder",
            display_name="Hy-Nomas Load Encoder",
            category=CATEGORY,
            description="Loads the DCAE-V video encoder (images -> the DiT's latents), for Start From Image. The "
                        "PixelUMM decoder file carries it, so pick the same file as Load Decoder.",
            inputs=[
                IO.Combo.Input("encoder_name", options=_encoder_files() or [""],
                               tooltip="The PixelUMM file (VAE/PixelUMM_FT_FP8, its encoder.* tensors) or the "
                                       "DCAE-V checkpoint (VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt)"),
            ],
            outputs=[ENCODER.Output(display_name="encoder")],
        )

    @classmethod
    def execute(cls, encoder_name) -> IO.NodeOutput:
        return IO.NodeOutput(dcae.load_encoder(folder_paths.get_full_path_or_raise("vae", encoder_name)))


class HyNomasStartFromImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasStartFromImage",
            display_name="Hy-Nomas Start From Image",
            category=CATEGORY,
            description="Starts a world from a picture: the image, held still, becomes segment 0 and the world "
                        "continues from it with segment 1. Hy-Nomas has no image input; this puts the image in its "
                        "memory of the past, which later segments continue (not trained for, but it holds the look "
                        "well). Segment 0 of the script should describe the image; its actions and times are not "
                        "used. Feed the world to Segment 1 or to Generate World.",
            inputs=[
                IO.Model.Input("model"),
                IO.Clip.Input("clip"),
                ENCODER.Input("encoder"),
                SCRIPT.Input("script"),
                IO.Image.Input("image", tooltip="Scaled to cover width x height and center-cropped"),
                IO.Int.Input("width", default=832, min=32, max=4096, step=32, tooltip=SIZE_TOOLTIP),
                IO.Int.Input("height", default=480, min=32, max=4096, step=32, tooltip=SIZE_TOOLTIP),
                IO.Int.Input("still_frames", default=3, min=1, max=10,
                             tooltip="Latent frames the image is held for (4 video frames each): the model's memory "
                                     "of it. More starts the motion more gently. Decode keeps one of these frames "
                                     "by default (its start_image option)."),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=False,
                             tooltip="Seed of the cache pass's re-noising"),
                IO.Int.Input("history_frames", default=18, min=0, max=1024, advanced=True, optional=True),
                IO.Boolean.Input("int8_kv", default=True, advanced=True, optional=True),
            ],
            outputs=[WORLD.Output(display_name="world"), IO.String.Output(display_name="summary")],
        )

    @classmethod
    def execute(cls, model, clip, encoder, script, image, width, height, still_frames, seed, history_frames=18,
                int8_kv=True) -> IO.NodeOutput:
        if len(script) < 2:
            raise ValueError("the script needs segment 0 for the image and at least one more segment to generate")
        world = session.start_world(model, clip, encoder, script.world, image, width, height,
                                    still_frames=still_frames, seed=seed, past_frames=history_frames,
                                    kv_int8=int8_kv)
        return IO.NodeOutput(world, world.describe())


class HyNomasLoadDecoder(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasLoadDecoder",
            display_name="Hy-Nomas Load Decoder",
            category=CATEGORY,
            description="Loads the PixelUMM video decoder (models/vae). Weights under NVIDIA's One-Way "
                        "Noncommercial license.",
            inputs=[
                IO.Combo.Input("decoder_name", options=_files("vae", "pixelumm") or [""],
                               tooltip="e.g. diffusion_pytorch_model_fp8.safetensors (VAE/PixelUMM_FT_FP8)"),
                IO.Combo.Input("matmul", options=["fp8", "dequant"], default="fp8",
                               tooltip="fp8: FP8 tensor cores (RTX 40). dequant: bf16 matmuls on the dequantized "
                                       "weights, closest to the trained decoder."),
            ],
            outputs=[DECODER.Output(display_name="decoder")],
        )

    @classmethod
    def execute(cls, decoder_name, matmul="fp8") -> IO.NodeOutput:
        path = folder_paths.get_full_path_or_raise("vae", decoder_name)
        return IO.NodeOutput(pixelumm.load_decoder(path, matmul=matmul))


class HyNomasDecode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasDecode",
            display_name="Hy-Nomas Decode",
            category=CATEGORY,
            description="Decodes a world (all its segments, one continuous clip) or a latent into frames.",
            inputs=[
                DECODER.Input("decoder"),
                WORLD.Input("world", optional=True),
                IO.Latent.Input("samples", optional=True, tooltip="Decoded after the world's segments, e.g. the "
                                                                  "last segment before it is added"),
                SCRIPT.Input("script", optional=True, tooltip="Trims to the script's frame count"),
                IO.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, control_after_generate=False,
                             advanced=True, optional=True, tooltip="The decoder's noise seed (Tencent's default 42)"),
                IO.Int.Input("window_overlap", default=5, min=1, max=9, advanced=True, optional=True,
                             tooltip="The decoder works on windows of 10 latent frames (40 video frames) that overlap "
                                     "by this many latent frames and are cross-faded. Tencent's 5 decodes every frame "
                                     "twice; 2 is ~1.6x faster with shorter cross-fades."),
                IO.Combo.Input("start_image", options=session.START_IMAGE_OPTIONS, default="one frame",
                               advanced=True, optional=True,
                               tooltip="For a world from Start From Image: keep one frame of the held image (the "
                                       "video opens on it and moves at once), the whole hold, or none."),
            ],
            outputs=[IO.Image.Output(display_name="frames"), IO.Float.Output(display_name="fps")],
        )

    @classmethod
    def execute(cls, decoder, world=None, samples=None, script=None, seed=42, window_overlap=5,
                start_image="one frame") -> IO.NodeOutput:
        parts = list(world.latents) if world is not None else []
        if samples is not None:
            parts.append(samples["samples"].float().cpu())
        if not parts:
            raise ValueError("connect a world or a latent")
        latent = torch.cat(parts, dim=2)
        frames = script.num_frames if script is not None and script.latent_boundaries[-1] == latent.shape[2] else None
        bar = comfy.utils.ProgressBar(1)

        def progress(done, total):
            bar.update_absolute(done, total)
        images = pixelumm.decode(decoder, latent, seed=seed, pixel_frames=frames, progress=progress,
                                 overlap=window_overlap)
        images = session.trim_start(images, world, start_image)
        return IO.NodeOutput(images, float(script.fps if script is not None else 24))


class HyNomasGenerateWorld(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasGenerateWorld",
            display_name="Hy-Nomas Generate World",
            category=CATEGORY,
            description="Every segment of a script in one node: Segment, sample (ComfyUI's sampler), Extend World, "
                        "in order. The same steps as the per-segment graph.",
            inputs=[
                IO.Model.Input("model"),
                IO.Clip.Input("clip"),
                SCRIPT.Input("script"),
                IO.Int.Input("width", default=832, min=32, max=4096, step=32, tooltip=SIZE_TOOLTIP),
                IO.Int.Input("height", default=480, min=32, max=4096, step=32, tooltip=SIZE_TOOLTIP),
                IO.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff),
                IO.Int.Input("steps", default=8, min=1, max=100),
                IO.Float.Input("cfg", default=1.0, min=0.0, max=20.0, step=0.1),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                WORLD.Input("world", optional=True, tooltip="Continue this world instead of starting a new one, e.g. "
                                                           "the one from Start From Image"),
                CAMERA.Input("camera", optional=True),
                IO.Int.Input("history_frames", default=18, min=0, max=1024, advanced=True, optional=True),
                IO.Boolean.Input("int8_kv", default=True, advanced=True, optional=True),
            ],
            outputs=[WORLD.Output(display_name="world")],
        )

    @classmethod
    def execute(cls, model, clip, script, width, height, seed, steps, cfg, sampler_name, scheduler, world=None,
                camera=None, history_frames=18, int8_kv=True) -> IO.NodeOutput:
        first = len(world.records) if world is not None else 0
        for index in range(first, len(script)):
            positive, latent = session.segment_conditioning(clip, script, index, width, height, world=world,
                                                            camera=camera)
            negative = session.negative_conditioning(clip, positive)
            segment_seed = seed + index
            noise = comfy.sample.prepare_noise(latent["samples"], segment_seed)
            samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                                          latent["samples"], seed=segment_seed,
                                          callback=None, disable_pbar=False)
            world = session.extend_world(model, positive, {"samples": samples}, seed=segment_seed,
                                         past_frames=history_frames, kv_int8=int8_kv)
        return IO.NodeOutput(world)


class HyNomasCameraKeys(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasCameraKeys",
            display_name="Hy-Nomas Camera Keys",
            category=CATEGORY,
            description="A camera path for one segment, played on the keys of Tencent's web demo: W/A/S/D move, "
                        "Left/Right/Up/Down turn (20 degrees/s), Shift runs. The path is stretched over the "
                        "segment it is connected to.",
            inputs=[
                IO.String.Input("timeline", multiline=True, default="W 2, W+Left 1.5, idle 0.5",
                                tooltip="Steps 'KEYS SECONDS', separated by commas or new lines. KEYS: W, A, S, D, "
                                        "Left, Right, Up, Down, Shift, joined by '+', or 'idle'."),
                IO.Float.Input("translation_scale", default=1.0, min=0.05, max=10.0, step=0.05,
                               tooltip="Gain on the (normalized) camera movement; Tencent's default 1.0."),
                IO.Float.Input("rotation_scale", default=2.5, min=0.05, max=10.0, step=0.05,
                               tooltip="Gain on the camera's turning; Tencent's default 2.5."),
            ],
            outputs=[CAMERA.Output(display_name="camera")],
        )

    @classmethod
    def execute(cls, timeline, translation_scale, rotation_scale) -> IO.NodeOutput:
        recording = cam.simulate_keys(cam.parse_timeline(timeline))
        return IO.NodeOutput(cam.CameraPath(recording, translation_scale, rotation_scale, timeline))


class HyNomasCameraRecording(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasCameraRecording",
            display_name="Hy-Nomas Camera Recording",
            category=CATEGORY,
            description="Draw a camera path for one segment: press Record in the node, then fly with W/A/S/D, the "
                        "arrow keys (or drag) and Shift, the controls of Tencent's web demo. A path recorded in the "
                        "demo (its JSON with a 'frames' list) can be pasted under { } JSON.",
            inputs=[
                IO.String.Input("recording_json", multiline=True, default='{"frames": []}',
                                tooltip="The recorded path: {frames: [{position, quaternion, fov}]} at 24 fps. "
                                        "Written by the recorder above."),
                IO.Float.Input("translation_scale", default=1.0, min=0.05, max=10.0, step=0.05),
                IO.Float.Input("rotation_scale", default=2.5, min=0.05, max=10.0, step=0.05),
            ],
            outputs=[CAMERA.Output(display_name="camera")],
        )

    @classmethod
    def execute(cls, recording_json, translation_scale, rotation_scale) -> IO.NodeOutput:
        import json
        recording = json.loads(recording_json or "{}")
        if len(recording.get("frames") or []) < 2:
            raise ValueError("Hy-Nomas Camera Recording: no path yet. Press Record in the node, fly with "
                             "W/A/S/D and the arrow keys, then Stop.")
        return IO.NodeOutput(cam.CameraPath(recording, translation_scale, rotation_scale, "recording"))


class HyNomasDrive(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HyNomasDrive",
            display_name="Hy-Nomas Drive",
            category=CATEGORY,
            description="Explore a world interactively: steer with W/A/S/D and the arrow keys in the node while it "
                        "generates turn by turn, each turn continuing the world with the keys you held since the "
                        "previous one. Use the panel's buttons; Queue/Run is not needed. Slower than real time: on "
                        "an RTX 4090 a 1 s turn takes about 3 s at 832x480.",
            inputs=[
                IO.Model.Input("model"),
                IO.Clip.Input("clip", tooltip="Qwen2.5-VL-7B in Load CLIP with type qwen_image"),
                DECODER.Input("decoder"),
                IO.String.Input("world_json", multiline=True, default=DEFAULT_PROMPT,
                                tooltip="The world to start from; its first segment is the first turn. Later turns "
                                        "take their text from the panel."),
                IO.Int.Input("width", default=832, min=64, max=2048, step=32, tooltip=SIZE_TOOLTIP),
                IO.Int.Input("height", default=480, min=64, max=2048, step=32, tooltip=SIZE_TOOLTIP),
                IO.Float.Input("turn_seconds", default=1.0, min=0.5, max=5.0, step=0.5,
                               tooltip="Video per turn. Shorter turns answer the keys sooner; longer ones are more "
                                       "efficient."),
                IO.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, control_after_generate=False),
                IO.Int.Input("history_frames", default=18, min=0, max=1024, advanced=True, optional=True),
                ENCODER.Input("encoder", optional=True, tooltip="With start_image: Load Encoder's output"),
                IO.Image.Input("start_image", optional=True,
                               tooltip="Start the drive from this picture (with an encoder): it becomes the world's "
                                       "first segment, described by world_json's first segment, and the first "
                                       "turn moves on from it"),
                IO.String.Input("command", default="", optional=True, advanced=True,
                                tooltip="Set by the drive panel."),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, clip, decoder, world_json, width, height, turn_seconds, seed, history_frames=18,
                encoder=None, start_image=None, command="") -> IO.NodeOutput:
        import json
        if not command:
            return IO.NodeOutput()
        cmd = json.loads(command)
        session_id = cmd["session"]
        if cmd.get("action") == "reset":
            drive.drop_session(session_id)
            return IO.NodeOutput()
        if (encoder is None) != (start_image is None):
            raise ValueError("to start from an image, connect both encoder and start_image")
        start = drive.StartImage(start_image, encoder) if start_image is not None else None
        config = (world_json, width, height, turn_seconds, seed, history_frames, start.key if start else None)
        session = drive.get_session(session_id, config, fresh=bool(cmd.get("fresh")), start=start)
        if cmd.get("action") == "save":
            session.export(decoder)
        else:
            session.run_turn(model, clip, decoder, timeline=cmd.get("timeline", ""),
                             steer=cmd.get("camera", "steer") == "steer", description=cmd.get("description", ""),
                             action=cmd.get("subject_action", ""))
        return IO.NodeOutput()


class HyNomasExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [HyNomasLoadDiT, HyNomasWorldPrompt, HyNomasSegment, HyNomasExtendWorld, HyNomasLoadEncoder,
                HyNomasStartFromImage, HyNomasLoadDecoder, HyNomasDecode, HyNomasGenerateWorld, HyNomasCameraKeys,
                HyNomasCameraRecording, HyNomasDrive]


async def comfy_entrypoint() -> HyNomasExtension:
    return HyNomasExtension()
