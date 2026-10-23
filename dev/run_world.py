"""Generate a world through the server with the per-segment graph (Segment -> KSampler -> Extend World) and save
an mp4. Usage:
    python dev/run_world.py prompt.json --width 832 --height 480 [--segments N] [--seconds S] [--matmul int8]
                                        [--image picture.png --still 3]
`--seconds S` rewrites the script's times to S-second segments (quick tests). `--image` starts the world from a
picture (Start From Image as segment 0; the file must be in the server's input folder)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api import Client  # noqa: E402


def retime(world, seconds, count):
    segs = world["segments"][:count] if count else world["segments"]
    t = 0.0
    for seg in segs:
        start, end = seg.get("start_time", "00:00.000"), seg.get("end_time", "00:00.000")

        def sec(ts):
            m, s = ts.split(":")
            return int(m) * 60 + float(s)
        old0, old1 = sec(start), sec(end)
        scale = seconds / max(old1 - old0, 1e-6)
        for act in seg.get("contained_actions", []) or []:
            for key in ("start_time", "end_time"):
                rel = (sec(act[key]) - old0) * scale
                act[key] = f"{int((t + rel) // 60):02d}:{(t + rel) % 60:06.3f}"
        seg["start_time"] = f"{int(t // 60):02d}:{t % 60:06.3f}"
        seg["end_time"] = f"{int((t + seconds) // 60):02d}:{(t + seconds) % 60:06.3f}"
        t += seconds
    world["segments"] = segs
    return world


def graph(world_json, n, width, height, seed, matmul, decoder_matmul, clip, prefix, steps=8, image=None, still=3):
    g = {
        "dit": {"class_type": "HyNomasLoadDiT", "inputs": {"dit_name": "hy_nomas_distil_int8.safetensors", "matmul": matmul}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image"}},
        "script": {"class_type": "HyNomasWorldPrompt", "inputs": {"world_json": world_json}},
        "dec": {"class_type": "HyNomasLoadDecoder", "inputs": {"decoder_name": "hy_nomas_pixelumm_fp8.safetensors",
                                                               "matmul": decoder_matmul}},
    }
    world = None
    first = 0
    if image:
        g["img"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g["enc"] = {"class_type": "HyNomasLoadEncoder", "inputs": {"encoder_name": "hy_nomas_pixelumm_fp8.safetensors"}}
        g["start"] = {"class_type": "HyNomasStartFromImage", "inputs": {
            "model": ["dit", 0], "clip": ["clip", 0], "encoder": ["enc", 0], "script": ["script", 0],
            "image": ["img", 0], "width": width, "height": height, "still_frames": still, "seed": seed}}
        world, first = "start", 1
    for i in range(first, n):
        seg = {"clip": ["clip", 0], "script": ["script", 0], "segment_index": i, "width": width, "height": height}
        if world:
            seg["world"] = [world, 0]
        g[f"seg{i}"] = {"class_type": "HyNomasSegment", "inputs": seg}
        g[f"ks{i}"] = {"class_type": "KSampler", "inputs": {
            "model": ["dit", 0], "seed": seed + i, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "positive": [f"seg{i}", 0], "negative": [f"seg{i}", 1],
            "latent_image": [f"seg{i}", 2], "denoise": 1.0}}
        g[f"ext{i}"] = {"class_type": "HyNomasExtendWorld", "inputs": {
            "model": ["dit", 0], "positive": [f"seg{i}", 0], "samples": [f"ks{i}", 0], "seed": seed + i}}
        world = f"ext{i}"
    g["decode"] = {"class_type": "HyNomasDecode", "inputs": {"decoder": ["dec", 0], "world": [world, 0],
                                                             "script": ["script", 0]}}
    g["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": ["decode", 1]}}
    g["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix,
                                                       "format": "mp4"}}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--segments", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--matmul", default="int8")
    ap.add_argument("--decoder-matmul", default="fp8")
    ap.add_argument("--clip", default="qwen_2.5_vl_7b_fp8_scaled.safetensors")
    ap.add_argument("--prefix", default="hynomas/test")
    ap.add_argument("--host", default="127.0.0.1:8199")
    ap.add_argument("--image", default=None)
    ap.add_argument("--still", type=int, default=3)
    args = ap.parse_args()
    data = json.loads(Path(args.prompt).read_text())
    world = data[0] if isinstance(data, list) else data
    if args.seconds:
        world = retime(world, args.seconds, args.segments)
    elif args.segments:
        world["segments"] = world["segments"][:args.segments]
    n = len(world["segments"])
    g = graph(json.dumps(world, ensure_ascii=False), n, args.width, args.height, args.seed, args.matmul,
              args.decoder_matmul, args.clip, args.prefix, image=args.image, still=args.still)
    history, timings, wall = Client(args.host).run(g)
    print(f"wall {wall:.1f}s")
    for node, sec in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  {node:8s} {g[node]['class_type']:22s} {sec:7.1f}s")
    for node, out in history.get("outputs", {}).items():
        for item in out.get("images", []) + out.get("videos", []) + out.get("gifs", []):
            print("output", item)


if __name__ == "__main__":
    main()
