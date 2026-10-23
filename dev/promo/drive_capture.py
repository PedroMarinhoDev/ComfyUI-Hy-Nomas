"""Drive mode for the promo: six 1 s turns from a start image through the Drive node (API), then Save mp4.

    python dev/promo/drive_capture.py paris_start.jpeg [--seed 7]    # the image must be in the server's input folder

Writes the turn keys to dev/promo/out/drive_keys.json; the video lands in the server's output/hy_nomas/drive_*.mp4."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from api import Client  # noqa: E402

TURNS = ["W 1", "W+Left 1", "W 1", "W+Right 1", "W 1", "W 1"]
ACTION = "walks forward along the street with a steady stride"


def graph(world, image, cmd, seed):
    return {
        "dit": {"class_type": "HyNomasLoadDiT", "inputs": {"dit_name": "hy_nomas_distil_int8.safetensors", "matmul": "int8"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "dec": {"class_type": "HyNomasLoadDecoder", "inputs": {"decoder_name": "hy_nomas_pixelumm_fp8.safetensors", "matmul": "fp8"}},
        "enc": {"class_type": "HyNomasLoadEncoder", "inputs": {"encoder_name": "hy_nomas_pixelumm_fp8.safetensors"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image}},
        "drive": {"class_type": "HyNomasDrive", "inputs": {
            "model": ["dit", 0], "clip": ["clip", 0], "decoder": ["dec", 0], "world_json": world, "width": 832,
            "height": 480, "turn_seconds": 1.0, "seed": seed, "encoder": ["enc", 0], "start_image": ["img", 0],
            "command": json.dumps(cmd)}}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    image = args.image
    world = (HERE.parents[1] / "examples/paris_from_image.json").read_text()
    client = Client()
    session = f"promo{int(time.time())}"
    timings = []
    for n, keys in enumerate(TURNS):
        cmd = dict(session=session, action="turn", fresh=n == 0, timeline=keys, camera="steer", description="",
                   subject_action=ACTION, nonce=n)
        _, _, wall = client.run(graph(world, image, cmd, args.seed), verbose=False)
        timings.append(round(wall, 2))
        print(f"turn {n} ({keys}) {wall:.1f}s", flush=True)
    history, _, _ = client.run(graph(world, image, dict(session=session, action="save", nonce=99), args.seed), verbose=False)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / "drive_keys.json").write_text(json.dumps({"turns": TURNS, "turn_seconds": 1.0, "wall": timings, "seed": args.seed,
                                                    "action": ACTION}, indent=1))
    print("saved, seed", args.seed)


if __name__ == "__main__":
    main()
