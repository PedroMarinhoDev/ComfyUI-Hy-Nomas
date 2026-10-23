"""camera.py's turn conversion against Tencent's demo_precomputed.prepare_turn + rotation gain (ref-venv)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "AI/research/hy-nomas-src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hy_nomas"))
import camera as C  # noqa: E402
from hymm.models.camera.geometry.demo_precomputed import prepare_turn, NORMALIZATION  # noqa: E402
from hymm.models.camera.geometry.rotation_scale import scale_framewise_rotations  # noqa: E402

for timeline, frames in (("W 2", 48), ("W+Left 1.5, idle 0.5, D+Up 1", 120), ("Left 3", 30), ("Shift+S 0.5, A 1", 72)):
    rec = C.simulate_keys(C.parse_timeline(timeline))
    ref = prepare_turn(rec, frames, NORMALIZATION)["norm_extrinsics"]
    ref = scale_framewise_rotations(ref, 2.5)
    got = C.turn_extrinsics(rec, frames, 1.0, 2.5)
    print(f"{timeline!r:32s} frames {frames}: poses {got.shape} max |diff| {np.abs(got - ref).max():.2e}")
