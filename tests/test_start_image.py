"""Start From Image: the image segment's prompt, trimming the hold, the drive's image key."""
import json
import sys
from pathlib import Path

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))


def test_image_prompt_drops_actions():
    from hy_nomas import prompt as P, session
    item = P.parse_world((PACK / "examples/paris_walk.json").read_text())
    prompt = json.loads(session.image_prompt(item))
    seg = prompt["segments"][0]
    assert prompt["Style"] == "video game" and "visual_description_long" in seg
    assert seg["contained_actions"] == []
    assert item["segments"][0]["contained_actions"]          # the script itself is left alone


def test_trim_start():
    from hy_nomas import session, world as W
    images = torch.arange(20.0).view(20, 1, 1, 1)
    world = W.World(start_frames=3)                          # 12 held video frames
    assert session.trim_start(images, world)[0].item() == 11
    assert len(session.trim_start(images, world, "none")) == 8
    assert len(session.trim_start(images, world, "whole hold")) == 20
    assert session.trim_start(images, W.World()) is images
    assert session.trim_start(images, None) is images


def test_still_clip_covers_and_crops():
    from hy_nomas import session
    image = torch.rand(1, 300, 600, 3)
    clip = session.still_clip(image, 256, 160, 8)
    assert clip.shape == (8, 160, 256, 3) and torch.equal(clip[0], clip[7])


def test_drive_start_image_key():
    from hy_nomas import drive
    a, b = torch.rand(1, 64, 64, 3), torch.rand(1, 64, 64, 3)
    assert drive.StartImage(a, None).key == drive.StartImage(a.clone(), None).key
    assert drive.StartImage(a, None).key != drive.StartImage(b, None).key
