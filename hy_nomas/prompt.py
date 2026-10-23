"""Hy-Nomas prompts: the structured JSON world description, cut into one text per segment.

Hy-Nomas was trained on JSON captions. A world is one JSON object (README "Prepare Prompts"): global fields
(`Style`, `perspective_view`, `subject_list`) and a list of `segments`, each with `start_time`/`end_time`, a
`visual_description_long` (first segment) or `description_change` (later ones) and optional `contained_actions`
with their own times. The reference turns it into one prompt per segment with the shaping rules of
`inference.yaml` (`keep_keywords`, `global_only_in_first_segment`, `no_desc_long_after_first`,
`json_prompt_normalize_action_timestamp`); the functions marked "vendored" below are copied from
`hymm/inference.py` (Tencent, Apache-2.0) so the strings come out byte for byte the same.

Each segment's text then goes through Qwen2.5-VL's chat template with the video system prompt
(`configs/constants.py`, `li-dit-encode-video-json`). The DiT reads the hidden states from the user turn on
(`crop_start`, 108 tokens): the user text plus `<|im_end|>\n<|im_start|>assistant\n`.

Two token offsets per segment drive the attention masks (`hymm/pipeline/base.py`):
* `action_start`: where `contained_actions` begins; later segments see only the text before it;
* `action_offsets` [o_0 .. o_A] and `action_windows` [[start, end, action]] (latent frames, segment-relative):
  within the segment, each action's video only sees its own action's text.
Both come from token-level common prefixes of the full text and its action-stripped variants, as there.
"""
import copy
import json
import re

PROMPT_SEPARATOR = "|||"
FPS = 24
TEXT_LEN = 1280

VIDEO_SYSTEM_PROMPT = (
    "You are a helpful assistant. Describe the video by detailing the following aspects:         1. The main "
    "content and theme of the video.         2. The color, shape, size, texture, quantity, text, and spatial "
    "relationships of the objects.         3. Actions, events, behaviors temporal relationships, physical movement "
    "changes of the objects.         4. background environment, light, style and atmosphere.         5. camera "
    "angles, movements, and transitions used in the video."
)
CHAT_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
USER_MARKER = "<|im_start|>user\n"

KEEP_KEYWORDS = [
    "Style", "perspective_view", "subject_list", "subject_list.subject_id", "subject_list.display_name",
    "subject_list.subject_attribute", "segments", "segments.visual_description_long", "segments.description_change",
    "segments.contained_actions", "segments.contained_actions.start_time", "segments.contained_actions.end_time",
    "segments.contained_actions.action_description",
]

_CAMERA_TEXT_BLOCK_RE = re.compile(r"\s*<Camera(\d+)>.*?</Camera\1>\s*", flags=re.DOTALL)
_BENCH_META_TOP_LEVEL_KEYS = (
    "test_id", "test_type", "test_focus", "dimension", "benchmark_kind", "trajectory_family", "trajectory_class",
    "trajectory_variant", "duration_bucket", "total_duration", "perspective", "perspective_compatibility",
    "scene_type", "scene_category", "content_domain", "game_genre", "obstacle_meta", "camera_trajectory_spec",
    "joint_control_spec", "evaluation_criteria",
)
_BENCH_META_SEGMENT_KEYS = ("segment_index", "segment_control", "joint_caption_constraint")
_BENCH_STRIP_KEYS = _BENCH_META_TOP_LEVEL_KEYS + _BENCH_META_SEGMENT_KEYS


# ---------------------------------------------------------------------------------------------------------------
# vendored from hymm/inference.py


def parse_timestamp(ts):
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(ts)


def _remove_keys_from_dict(d, keys_to_remove):
    tree = {}
    global_keys = set()
    for key in keys_to_remove:
        parts = key.split(".")
        if len(parts) == 1:
            global_keys.add(parts[0])
        else:
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node.setdefault(None, set()).add(parts[-1])
    tree[None] = global_keys

    def _apply(obj, t, inherited_globals):
        local_keys = t.get(None, set())
        remove_here = local_keys | inherited_globals
        result = {}
        for k, v in obj.items():
            if k in remove_here:
                continue
            sub_tree = t.get(k)
            if isinstance(v, list):
                child_tree = {kk: vv for kk, vv in t.items() if kk is not None}
                if sub_tree:
                    child_tree.update(sub_tree)
                v = [_apply(item, child_tree, inherited_globals) if isinstance(item, dict) else item for item in v]
            result[k] = v
        return result

    return _apply(copy.deepcopy(d), tree, global_keys)


def _keep_keys_in_dict(d, keys_to_keep):
    legacy_alias = {"events": "segments", "subjects_lifecycle": "subject_list"}
    tree = {}
    if keys_to_keep:
        for key in keys_to_keep:
            parts = key.split(".")
            if parts:
                parts[0] = legacy_alias.get(parts[0], parts[0])
            if len(parts) == 1:
                tree.setdefault(None, set()).add(parts[0])
            else:
                node = tree
                for part in parts[:-1]:
                    node.setdefault(None, set()).add(part)
                    node = node.setdefault(part, {})
                node.setdefault(None, set()).add(parts[-1])

    def _apply(obj, t, inherited_globals):
        local_keys = t.get(None, set()) if t else set()
        keep_here = local_keys | inherited_globals
        result = {}
        for k, v in obj.items():
            if k not in keep_here:
                continue
            sub_tree = t.get(k) if t else None
            if sub_tree:
                if isinstance(v, dict):
                    v = _apply(v, sub_tree, inherited_globals)
                elif isinstance(v, list):
                    v = [_apply(item, sub_tree, inherited_globals) if isinstance(item, dict) else item for item in v]
            result[k] = v
        return result

    inherited_globals = tree.get(None, set()) if tree else set()
    return _apply(copy.deepcopy(d), tree, inherited_globals)


def _format_relative_timestamp(seconds):
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:06.3f}"


def _normalize_contained_action_timestamps(obj):
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list):
        return obj
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        actions = seg.get("contained_actions")
        if not isinstance(actions, list):
            continue
        segment_start = parse_timestamp(seg.get("start_time", 0.0))
        normalized = []
        for action in actions:
            if isinstance(action, dict):
                action_out = dict(action)
                for key in ("start_time", "end_time"):
                    if key in action_out:
                        rel_t = parse_timestamp(action_out[key]) - segment_start
                        action_out[key] = _format_relative_timestamp(rel_t)
            else:
                action_out = action
            normalized.append(action_out)
        seg["contained_actions"] = normalized
    return obj


def _strip_camera_text_blocks_for_camera_segment(seg):
    if not isinstance(seg, dict) or seg.get("segment_control") != "camera":
        return seg

    def _clean(value):
        if isinstance(value, str):
            if "<Camera" not in value:
                return value
            cleaned = _CAMERA_TEXT_BLOCK_RE.sub(" ", value)
            return re.sub(r"\s+", " ", cleaned).strip()
        if isinstance(value, dict):
            return {key: _clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_clean(item) for item in value]
        return value

    return _clean(seg)


def _canonicalize_stage2_keys(item):
    if not isinstance(item, dict):
        return item
    if "segments" in item and "subject_list" in item:
        return item
    out = dict(item)
    if "segments" not in out and "events" in out:
        out["segments"] = out.pop("events")
    if "subject_list" not in out and "subjects_lifecycle" in out:
        out["subject_list"] = out.pop("subjects_lifecycle")
    return out


def _build_whole_item_prompts(item, segments, keep_keywords=KEEP_KEYWORDS, global_only_in_first_segment=True,
                              no_desc_long_after_first=True, normalize_action_timestamp=True):
    if keep_keywords:
        cleaned = _keep_keys_in_dict(item, list(keep_keywords))
    else:
        cleaned = copy.deepcopy(item)
    cleaned = _remove_keys_from_dict(cleaned, list(_BENCH_STRIP_KEYS))
    context = {k: v for k, v in cleaned.items() if k != "segments"}
    per_segment = []
    for i, seg in enumerate(segments):
        seg = _strip_camera_text_blocks_for_camera_segment(seg)
        seg_processed = dict(seg)
        seg_idx = seg_processed.get("_segment_idx", i)
        seg_processed.pop("_segment_idx", None)
        if no_desc_long_after_first:
            if i > 0:
                seg_processed.pop("visual_description_long", None)
            if "description_change" in seg_processed and seg_idx == 0:
                seg_processed.pop("description_change", None)
        if "contained_actions" in seg_processed:
            actions = seg_processed.pop("contained_actions")
            seg_processed["contained_actions"] = actions
        if global_only_in_first_segment and i > 0:
            segment_obj = {"segments": [seg_processed]}
        else:
            segment_obj = {**context, "segments": [seg_processed]}
        if normalize_action_timestamp:
            segment_obj = _normalize_contained_action_timestamps(segment_obj)
        if keep_keywords:
            segment_obj = _keep_keys_in_dict(segment_obj, list(keep_keywords))
        per_segment.append(json.dumps(segment_obj, ensure_ascii=False))
    return per_segment


# ---------------------------------------------------------------------------------------------------------------
# the world script


def parse_world(text):
    """A world JSON (an object, or a list holding one object, as in Tencent's prompts.json) -> dict."""
    data = json.loads(text)
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"the prompt JSON holds {len(data)} worlds; give one")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("the prompt JSON must be an object with a 'segments' list")
    data = _canonicalize_stage2_keys(data)
    if not isinstance(data.get("segments"), list) or not data["segments"]:
        raise ValueError("the prompt JSON needs a non-empty 'segments' list")
    return data


def segment_prompts(world):
    """One JSON string per segment, shaped as the reference shapes them."""
    return _build_whole_item_prompts(world, world["segments"])


def pixel_boundaries(world, fps=FPS):
    """Segment boundaries in video frames, `int(seconds * fps)` (`_load_from_json`), or None without times."""
    out = []
    for seg in world["segments"]:
        st, et = seg.get("start_time"), seg.get("end_time")
        if st is None or et is None:
            return None
        if not out:
            out.append(int(parse_timestamp(st) * fps))
        out.append(int(parse_timestamp(et) * fps))
    return out


def latent_boundaries(pixel_bounds):
    """Pixel boundaries -> latent frames, as `_process_single_prompt` converts them: the video has
    (frames - 1) // 4 + 1 latents and boundary b maps to b // 4, kept strictly increasing."""
    num_frames = pixel_bounds[-1]
    latent_frames = (num_frames - 1) // 4 + 1
    out = [0]
    for b in pixel_bounds[1:]:
        out.append(max(out[-1] + 1, min(b // 4, latent_frames)))
    out[-1] = latent_frames
    return out


# ---------------------------------------------------------------------------------------------------------------
# tokens


class PromptTokenizer:
    """Qwen2.5-VL's tokenizer with the video template; the ids match `TextEncoder.text2tokens` for one prompt."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer          # a Hugging Face Qwen2 tokenizer (ComfyUI ships its files)
        self.crop_start = self._crop_start()

    def ids(self, text, max_text=TEXT_LEN):
        """Token ids of the full chat for `text` (the reference truncates at crop_start + text_len)."""
        full = CHAT_TEMPLATE.format(system=VIDEO_SYSTEM_PROMPT, text=text if text else " ")
        ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        return ids[: self.crop_start + max_text]

    def _crop_start(self):
        ids = self.tokenizer(CHAT_TEMPLATE.format(system=VIDEO_SYSTEM_PROMPT, text="a photo of a cat"),
                             add_special_tokens=False)["input_ids"]
        marker = self.tokenizer(USER_MARKER, add_special_tokens=False)["input_ids"]
        for i in range(len(ids) - len(marker) + 1):
            if ids[i:i + len(marker)] == marker:
                return i + len(marker)
        raise ValueError("the chat template has no user turn")

    def _common_prefix_offset(self, full_ids, variant, max_text=TEXT_LEN):
        v_ids = self.ids(variant, max_text)
        n = min(len(full_ids), len(v_ids))
        common = 0
        while common < n and full_ids[common] == v_ids[common]:
            common += 1
        return max(0, min(common - self.crop_start, max_text - 1))

    def action_start(self, prompt):
        """`_compute_action_start`: -1 when the segment has no actions."""
        try:
            obj = json.loads(prompt)
        except Exception:
            return -1
        segs = obj.get("segments") if isinstance(obj, dict) else None
        if not (isinstance(segs, list) and segs and isinstance(segs[0], dict)):
            return -1
        acts = segs[0].get("contained_actions")
        if not ((isinstance(acts, list) and acts) or (isinstance(acts, dict) and acts)):
            return -1
        strip_obj = {k: ([{ek: sv for ek, sv in segs[0].items() if ek != "contained_actions"}] if k == "segments" else v)
                     for k, v in obj.items()}
        return self._common_prefix_offset(self.ids(prompt), json.dumps(strip_obj, ensure_ascii=False))

    def action_offsets(self, prompt):
        """`_compute_action_text_offsets`: [o_0, ..., o_A] in cropped-token coordinates, or None.

        The last entry is the reference's `min(valid, text_len)` over the uncropped sequence; the attention clamps
        every offset to the number of text tokens, which makes it the end of the text."""
        try:
            obj = json.loads(prompt)
        except Exception:
            return None
        segs = obj.get("segments") if isinstance(obj, dict) else None
        if not (isinstance(segs, list) and segs and isinstance(segs[0], dict)):
            return None
        seg0 = segs[0]
        acts = seg0.get("contained_actions")
        if isinstance(acts, dict) and acts:
            act_items, num_actions = list(acts.items()), len(acts)
        elif isinstance(acts, list) and acts:
            act_items, num_actions = None, len(acts)
        else:
            return None

        def variant(k):
            seg_k = dict(seg0)
            if k <= 0:
                seg_k.pop("contained_actions", None)
            elif act_items is not None:
                seg_k["contained_actions"] = {key: val for key, val in act_items[:k]}
            else:
                seg_k["contained_actions"] = acts[:k]
            return json.dumps({key: ([seg_k] if key == "segments" else val) for key, val in obj.items()},
                              ensure_ascii=False)

        full_ids = self.ids(prompt)
        cleaned, prev = [], 0
        for k in range(num_actions):
            o = max(self._common_prefix_offset(full_ids, variant(k)), prev)
            o = min(o, TEXT_LEN)
            cleaned.append(o)
            prev = o
        cleaned.append(min(len(full_ids), TEXT_LEN))
        return cleaned


def action_windows(prompt, seg_frames, fps=FPS, t_ratio=4):
    """`_compute_action_video_windows`: [[start, end, action or -1], ...] tiling the segment's latent frames, or
    None when there is nothing to restrict (no timed actions, or one action covering the whole segment)."""
    if seg_frames <= 0:
        return None
    try:
        obj = json.loads(prompt)
    except Exception:
        return None
    segs = obj.get("segments") if isinstance(obj, dict) else None
    if not (isinstance(segs, list) and segs and isinstance(segs[0], dict)):
        return None
    acts = segs[0].get("contained_actions")
    if isinstance(acts, dict):
        action_list = list(acts.values())
    elif isinstance(acts, list):
        action_list = acts
    else:
        return None
    if not action_list:
        return None

    def to_latent(t):
        return int(max(t, 0.0) * fps) // int(t_ratio)

    spans = []
    for k, a in enumerate(action_list):
        if not isinstance(a, dict) or "start_time" not in a or "end_time" not in a:
            return None
        vs = max(0, min(to_latent(parse_timestamp(a["start_time"])), seg_frames))
        ve = max(0, min(to_latent(parse_timestamp(a["end_time"])), seg_frames))
        if ve > vs:
            spans.append([vs, ve, k])
    if not spans:
        return None
    spans.sort(key=lambda s: (s[0], s[1]))
    for p in range(1, len(spans)):
        if spans[p][0] < spans[p - 1][1]:
            raise ValueError(f"overlapping actions {spans[p - 1]} and {spans[p]} in one segment are not supported")
    windows, cursor = [], 0
    for vs, ve, k in spans:
        if vs > cursor:
            windows.append([cursor, vs, -1])
        windows.append([vs, ve, k])
        cursor = ve
    if cursor < seg_frames:
        windows.append([cursor, seg_frames, -1])
    if len(windows) == 1 and windows[0][2] == 0 and len(action_list) == 1:
        return None
    return windows
