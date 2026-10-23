"""Drive mode's per-turn prompts follow the demo's segment contract."""
import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))


def session():
    from hy_nomas import drive
    world = (PACK / "docs/upstream/example_prompts.json").read_text()
    return drive.DriveSession("t", world, 832, 480, 1.0, 42, 18)


def test_turn_prompts():
    from hy_nomas import drive
    s = session()
    first = json.loads(s.turn_prompt(0))
    assert first["Style"] == "Cinematic fantasy"
    seg = first["segments"][0]
    assert "visual_description_long" in seg and "start_time" not in seg
    assert seg["contained_actions"][0]["end_time"] == "00:01.000"   # the 5 s action re-timed to the 1 s turn
    later = json.loads(s.turn_prompt(3))
    assert list(later) == ["segments"]
    assert later["segments"][0] == {"description_change": drive.FALLBACK, "contained_actions": []}
    said = json.loads(s.turn_prompt(2, "a red balloon floats up", "Jumps over the rock."))["segments"][0]
    assert said["description_change"] == "Compared to the previous segment, a red balloon floats up"
    assert said["contained_actions"] == [{"start_time": "00:00.000", "end_time": "00:01.000",
                                          "action_description": "<Action1>a small orange fox jumps over the rock</Action1>."}]
    assert s.turn_latents == 6


def test_action_sentence_trained_format():
    from hy_nomas import drive
    world = (PACK / "examples/paris_walk.json").read_text()
    s = drive.DriveSession("t", world, 832, 480, 1.0, 42, 18)
    seg = json.loads(s.turn_prompt(1, "", "stops and looks up at the tower"))["segments"][0]
    assert seg["contained_actions"][0]["action_description"] == \
        "<Action1><|Subject1|> stops and looks up at the tower</Action1>."
    tagged = "<Action1><|Subject1|> waves</Action1>."
    assert drive.action_sentence(tagged) == tagged
