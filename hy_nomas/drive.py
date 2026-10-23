"""Drive mode: a world generated turn by turn while you steer it, the way Tencent's web demo works.

Each turn is one short segment (1-3 s): its camera path comes from the keys you held since the previous turn was
sent (`camera.parse_timeline` / `simulate_keys`, the demo's controls), its text from the world prompt (first turn)
or from what you type (later turns), and it attends to everything generated before through the world's history.
After it is sampled and added to the world, only its new frames are decoded (`pixelumm.decode_new_frames`) and
sent to the browser as JPEGs over ComfyUI's websocket.

Later turns follow the demo's prompt contract (`apps/prompts/online_user_video_rewriter_*`): a follow-up segment
has a `description_change` starting "Compared to the previous segment, " and, when the camera is steered, no
actions; with nothing to say, the demo's fallback sentence is used. A typed action becomes the trained action
sentence, "<Action1><|Subject1|> turns around</Action1>." (`action_sentence`).

With a start image (`StartImage`), the world's first segment is that picture held still (`session.start_world`)
and the first turn is already a follow-up.

A session lives in this process (`SESSIONS`), keyed by an id the node's panel makes; the panel sends one command
per turn by queueing the drive node alone (`partial_execution_targets`), so the loaders above it stay cached.
"""
import base64
import hashlib
import io
import logging
import time

import torch
from PIL import Image

import comfy.model_management
import comfy.sample

from . import camera as cam
from . import pixelumm
from . import prompt as P
from . import session as S
from . import text as T
from . import world as W

FALLBACK = "Compared to the previous segment, the visible scene and composition transition continuously."
PREFIX = "Compared to the previous segment, "
SESSIONS = {}
MAX_SESSIONS = 4


def _fmt(seconds):
    return P._format_relative_timestamp(seconds)


def subject_token(item):
    """How actions name the subject: its `<|Subject1|>` id (the trained format), else its display name."""
    subjects = item.get("subject_list") or [{}]
    sid = str(subjects[0].get("subject_id", ""))
    return sid if sid.startswith("<|") else str(subjects[0].get("display_name") or "the subject")


def action_sentence(text, subject="<|Subject1|>"):
    """A typed action ("turns around and runs back") as the rewriter writes it; tagged text is kept as is."""
    text = text.strip()
    if not text or "<Action" in text:
        return text
    text = text.rstrip(".")
    return f"<Action1>{subject} {text[0].lower() + text[1:]}</Action1>."


def send(event, data):
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(event, data, PromptServer.instance.client_id)
    except Exception:  # outside a server (tests)
        logging.debug("Hy-Nomas drive: %s %s", event, {k: v for k, v in data.items() if k != "frames"})


def jpeg_frames(frames, quality=88):
    out = []
    for frame in frames.numpy():
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


class StartImage:
    """A picture to start from and the encoder for it; `key` tells pictures apart in the session config."""

    def __init__(self, image, encoder, still_frames=3):
        self.image, self.encoder, self.still_frames = image, encoder, still_frames
        small = image[:1, ::8, ::8].float().mul(255).round().to(torch.uint8).contiguous()
        self.key = (tuple(image.shape), hashlib.sha1(small.numpy().tobytes()).hexdigest()[:16], still_frames)


class DriveSession:
    def __init__(self, session_id, world_json, width, height, turn_seconds, seed, history_frames, start_key=None,
                 start=None):
        self.id = session_id
        self.item = P.parse_world(world_json)
        self.width, self.height = width, height
        self.turn_seconds = float(turn_seconds)
        self.turn_latents = int(round(self.turn_seconds * P.FPS)) // 4
        if self.turn_latents < 1:
            raise ValueError("turn length too short")
        self.seed = int(seed)
        self.history_frames = int(history_frames)
        self.world = None
        self.turns = []                 # per turn: text, timeline, seconds taken
        self.decoded_end = 0
        self._text_cache = {}
        self.start = start
        self.config = (world_json, width, height, turn_seconds, seed, history_frames, start_key)

    # -- prompts

    def turn_prompt(self, index, description="", action=""):
        """The DiT prompt string of turn `index`, shaped like any segment of a world (`prompt.py`)."""
        start, end = index * self.turn_seconds, (index + 1) * self.turn_seconds
        first = dict(self.item["segments"][0])
        old_start = P.parse_timestamp(first.get("start_time", 0))
        old_end = P.parse_timestamp(first.get("end_time", old_start + self.turn_seconds))
        scale = self.turn_seconds / max(old_end - old_start, 1e-6)
        if first.get("contained_actions"):
            first["contained_actions"] = [
                dict(a, start_time=_fmt((P.parse_timestamp(a.get("start_time", old_start)) - old_start) * scale),
                     end_time=_fmt((P.parse_timestamp(a.get("end_time", old_end)) - old_start) * scale))
                for a in first["contained_actions"]]
        first["start_time"], first["end_time"] = _fmt(0), _fmt(self.turn_seconds)
        if index == 0:
            if description:
                first["visual_description_long"] = description
            return P._build_whole_item_prompts(self.item, [first])[0]
        change = description.strip() or FALLBACK
        if not change.startswith(PREFIX):
            change = PREFIX + change[0].lower() + change[1:]
        seg = {"start_time": _fmt(start), "end_time": _fmt(end), "description_change": change,
               "contained_actions": ([{"start_time": _fmt(start), "end_time": _fmt(end),
                                       "action_description": action_sentence(action, subject_token(self.item))}]
                                     if action.strip() else [])}
        return P._build_whole_item_prompts(self.item, [first, seg])[1]

    def encode(self, clip, prompt):
        cached = self._text_cache.get(prompt)
        if cached is None:
            cached = T.encode_segment(clip, prompt, self.turn_latents)
            self._text_cache[prompt] = cached
        states, layout = cached
        return states, dict(layout)

    # -- one turn

    def run_turn(self, model, clip, decoder, timeline="", steer=True, description="", action=""):
        started = time.time()
        if self.world is None and self.start is not None:
            send("hynomas.drive", {"session": self.id, "type": "progress", "turn": 0, "stage": "start image"})
            self.world = S.start_world(model, clip, self.start.encoder, self.item, self.start.image, self.width,
                                       self.height, still_frames=self.start.still_frames, seed=self.seed,
                                       past_frames=self.history_frames)
            self.decoded_end = self.world.start_frames
        # with a start image, the image is segment 0 and every turn is a follow-up
        turn = len(self.turns)
        index = turn + (1 if self.start is not None else 0)
        prompt = self.turn_prompt(index, description, action)
        states, layout = self.encode(clip, prompt)
        layout.update(segment_index=index, frames=self.turn_latents, world=self.world)
        if steer:
            steps = cam.parse_timeline(timeline) if timeline.strip() else [(set(), self.turn_seconds)]
            layout["camera"] = cam.CameraPath(cam.simulate_keys(steps)).tensor(self.turn_latents)
        positive = [[states.to(torch.bfloat16), {"nomas": layout}]]
        latent = torch.zeros([1, 32, self.turn_latents, self.height // 32, self.width // 32])
        seed = self.seed + index
        noise = comfy.sample.prepare_noise(latent, seed)

        def callback(step, x0, x, total):
            send("hynomas.drive", {"session": self.id, "type": "progress", "turn": turn, "stage": "sampling",
                                   "step": step + 1, "steps": total})
        samples = comfy.sample.sample(model, noise, 8, 1.0, "euler", "simple", positive, positive, latent,
                                      seed=seed, callback=callback, disable_pbar=True)
        sampled = time.time()
        send("hynomas.drive", {"session": self.id, "type": "progress", "turn": turn, "stage": "memory"})
        self.world = S.extend_world(model, positive, {"samples": samples}, seed=seed,
                                    past_frames=self.history_frames, kv_int8=True)
        extended = time.time()
        send("hynomas.drive", {"session": self.id, "type": "progress", "turn": turn, "stage": "decoding"})
        timeline_latent = self.world.full_latent()
        end = timeline_latent.shape[2]
        frames = pixelumm.decode_new_frames(decoder, timeline_latent, self.decoded_end, end)
        self.decoded_end = end
        done = time.time()
        timing = {"sampling": round(sampled - started, 2), "memory": round(extended - sampled, 2),
                  "decoding": round(done - extended, 2), "total": round(done - started, 2)}
        self.turns.append({"prompt": prompt, "timeline": timeline, "steer": steer, "timing": timing})
        logging.info("Hy-Nomas drive %s: turn %d (%s) %s", self.id[:8], turn, timeline or "no keys", timing)
        send("hynomas.drive", {"session": self.id, "type": "frames", "turn": turn, "fps": P.FPS,
                               "frames": jpeg_frames(frames), "timing": timing, "seconds": self.seconds(),
                               "archive_gib": round(self.world.nbytes() / 2 ** 30, 2)})
        return frames

    def seconds(self):
        return len(self.turns) * self.turn_seconds

    def export(self, decoder, prefix="hy_nomas/drive"):
        """Decode the whole drive properly (overlapping windows, cross-faded) and save it as an mp4."""
        import folder_paths
        from comfy_api.latest import InputImpl, Types
        if self.world is None:
            raise ValueError("nothing to save yet")
        send("hynomas.drive", {"session": self.id, "type": "progress", "stage": "saving"})
        images = S.trim_start(pixelumm.decode(decoder, self.world.full_latent()), self.world)
        from fractions import Fraction
        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, frame_rate=Fraction(P.FPS)))
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1])
        file = f"{name}_{counter:05}_.mp4"
        import os
        video.save_to(os.path.join(folder, file), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
        info = {"filename": file, "subfolder": subfolder, "type": "output"}
        send("hynomas.drive", {"session": self.id, "type": "saved", "file": info, "seconds": self.seconds()})
        return info


def get_session(session_id, config, fresh=False, start=None):
    session = SESSIONS.get(session_id)
    if session is None or fresh or session.config != config:
        if session is not None:
            SESSIONS.pop(session_id, None)
        while len(SESSIONS) >= MAX_SESSIONS:
            SESSIONS.pop(next(iter(SESSIONS)))
        session = DriveSession(session_id, *config, start=start)
        SESSIONS[session_id] = session
    return session


def drop_session(session_id):
    SESSIONS.pop(session_id, None)


def register_routes():
    """POST /hynomas/drive/reset {"session": id}: drop a session's world (its memory) without queueing."""
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return

    @PromptServer.instance.routes.post("/hynomas/drive/reset")
    async def reset(request):
        data = await request.json()
        drop_session(str(data.get("session", "")))
        return web.json_response({"ok": True, "sessions": len(SESSIONS)})


register_routes()
