"""prompt.py against Tencent's own shaping, tokenization and action layout (dev/capture_prompts.py output)."""
import json
import os
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
ORACLE = Path(os.environ.get("HY_NOMAS_LAB", Path.home() / "AI/hy-nomas-lab")) / "captures/prompts_oracle.json"


def tokenizer():
    import comfy.text_encoders.qwen_image
    return comfy.text_encoders.qwen_image.QwenImageTokenizer().qwen25_7b.tokenizer


@pytest.mark.skipif(not ORACLE.exists(), reason="no oracle capture")
def test_against_reference():
    from hy_nomas import prompt as P
    oracle = json.loads(ORACLE.read_text())
    tok = P.PromptTokenizer(tokenizer())
    assert tok.crop_start == oracle["crop_start"]
    assert P.VIDEO_SYSTEM_PROMPT == oracle["system"]
    for case in oracle["cases"]:
        path = Path(case["file"])
        path = path if path.is_absolute() else ORACLE.parent.parent / path
        world = P.parse_world(path.read_text())
        prompts = P.segment_prompts(world)
        bounds = P.pixel_boundaries(world)
        assert bounds == case["pixel_boundaries"]
        lb = P.latent_boundaries(bounds)
        assert lb == case["latent_boundaries"]
        for i, (p, seg) in enumerate(zip(prompts, case["segments"])):
            assert p == seg["prompt"]
            assert tok.ids(p) == seg["ids"]
            assert tok.action_start(p) == seg["action_start"]
            offs = tok.action_offsets(p)
            assert offs == seg["offsets"]
            wins = P.action_windows(p, lb[i + 1] - lb[i]) if offs is not None else None
            assert wins == seg["windows"]
