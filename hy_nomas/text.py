"""Segment text -> the DiT's text conditioning, through ComfyUI's own Qwen2.5-VL-7B text encoder.

Tencent's `TextEncoder` (`hymm/models/text_encoder/encoder.py`, `inference.yaml`): the language model of
Qwen2.5-VL-7B-Instruct, the video chat template (`prompt.CHAT_TEMPLATE`), `hidden_state_skip_layer` 2 (that is
`hidden_states[-3]`, the output of layer 25 of 28, no final norm), cropped after `<|im_start|>user\n`, valid tokens
only, at most 1280.

The stock CLIP of type `qwen_image` (the Qwen-Image text encoder, same weights) does all of that except the
template and the layer: it crops at the same marker. So the ids are built here (`prompt.PromptTokenizer`) and
handed to it as tokens, with `clip_layer(-3)`.
"""
import torch

from . import prompt as P


def _tokenizer(clip):
    tok = getattr(clip.tokenizer, "qwen25_7b", None)
    model = getattr(clip.cond_stage_model, "qwen25_7b", None)
    if tok is None or model is None:
        raise ValueError("Hy-Nomas needs the Qwen2.5-VL-7B text encoder loaded with type 'qwen_image' "
                         "(e.g. qwen_2.5_vl_7b_fp8_scaled.safetensors in Load CLIP)")
    return tok.tokenizer


_TOKENIZERS = {}


def prompt_tokenizer(clip):
    hf = _tokenizer(clip)
    key = id(hf)
    if key not in _TOKENIZERS:
        _TOKENIZERS[key] = P.PromptTokenizer(hf)
    return _TOKENIZERS[key]


def encode_ids(clip, ids):
    """Token ids of the full chat -> hidden states of layer 25 from the user turn on, [1, L, 3584] float32."""
    clip = clip.clone()
    clip.clip_layer(-3)
    tokens = {"qwen25_7b": [[(int(i), 1.0) for i in ids]]}
    out = clip.encode_from_tokens(tokens, return_dict=True)
    cond = out.pop("cond")
    expected = len(ids) - prompt_tokenizer(clip).crop_start
    if cond.shape[1] != expected:
        raise RuntimeError(f"text encoder returned {cond.shape[1]} tokens, expected {expected}: is this the "
                           "qwen_image text encoder?")
    return cond


def encode_segment(clip, segment_prompt, seg_frames=None):
    """One segment's prompt string -> (text states [1, L, 3584], layout dict for `NomasCond`)."""
    tok = prompt_tokenizer(clip)
    ids = tok.ids(segment_prompt)
    states = encode_ids(clip, ids)
    layout = {"text": segment_prompt, "text_tokens": states.shape[1], "action_start": tok.action_start(segment_prompt)}
    offsets = tok.action_offsets(segment_prompt)
    windows = None
    if offsets is not None and seg_frames:
        windows = P.action_windows(segment_prompt, seg_frames)
        if windows is None:
            offsets = None
    layout["action_offsets"] = offsets
    layout["action_windows"] = windows
    return states, layout


def head_tokens(layout):
    """How many of a segment's text tokens later segments may see (`cross_segment_text_head_only`)."""
    start = layout.get("action_start", -1)
    total = layout["text_tokens"]
    return total if start is None or start < 0 else min(int(start), total)
