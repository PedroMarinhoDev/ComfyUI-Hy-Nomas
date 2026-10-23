"""Hy-Nomas on smaller GPUs, simulated on the 4090: a second process holds the VRAM a smaller card would not have,
and a fresh server (dev/serve.sh) is started per size so ComfyUI sizes itself against what is left.

    python dev/bench_lowvram.py --sizes 24,16,12 --res 1280x704 [--prompt ...] [--seconds 5]

Per size: the full graph twice (cold, then warm with another seed), per-node seconds, the card's peak use.
Results: ~/AI/hy-nomas-lab/bench/lowvram.jsonl.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from api import Client  # noqa: E402
from run_world import graph, retime  # noqa: E402

LAB = Path(os.environ.get("HY_NOMAS_LAB", Path.home() / "AI/hy-nomas-lab"))
GPU = "1"
FULL_MIB = 24564
PY = str(Path.home() / "AI/ComfyUI/.venv/bin/python")
FILLER = ("import sys, torch; block = torch.empty(int(sys.argv[1]) * 2**20, dtype=torch.uint8, device='cuda'); "
          "print('held', flush=True); sys.stdin.read()")


def used_mib():
    out = subprocess.run(["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"))
    return int(out.stdout.strip())


class Peak:
    def __enter__(self):
        self.peak, self.stop = 0, threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def loop(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, used_mib())
            time.sleep(0.2)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16,12")
    ap.add_argument("--res", default="1280x704")
    ap.add_argument("--prompt", default=str(Path.home() / "AI/hf_originals/Hy-Nomas/examples/prompts.json"))
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--port", type=int, default=8198)
    ap.add_argument("--matmul", default="int8")
    args = ap.parse_args()
    width, height = map(int, args.res.split("x"))
    data = json.loads(Path(args.prompt).read_text())
    world = data[0] if isinstance(data, list) else data
    if args.seconds:
        world = retime(world, args.seconds, 0)
    out_path = LAB / "bench/lowvram.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU)
    for size in [int(s) for s in args.sizes.split(",")]:
        idle = used_mib()
        hold = FULL_MIB - size * 1024 - idle - 300
        filler = None
        if hold > 0:
            filler = subprocess.Popen([PY, "-c", FILLER, str(hold)], env=env, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
            filler.stdout.readline()
        log = open(LAB / f"bench/server_{size}gb.log", "w")
        server = subprocess.Popen([str(HERE / "serve.sh")], env=dict(os.environ, PORT=str(args.port), GPU=GPU),
                                  stdout=log, stderr=subprocess.STDOUT)
        client = Client(f"127.0.0.1:{args.port}")
        try:
            for _ in range(600):
                try:
                    client._get("/system_stats")
                    break
                except Exception:
                    time.sleep(1)
            for run, seed in (("cold", 42), ("warm", 43)):
                g = graph(json.dumps(world, ensure_ascii=False), len(world["segments"]), width, height, seed,
                          args.matmul, "fp8", "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                          f"hynomas/lowvram_{size}gb_{run}")
                record = {"size_gb": size, "run": run, "res": args.res, "segments": len(world["segments"])}
                with Peak() as peak:
                    try:
                        _, timings, wall = client.run(g, verbose=False)
                        record.update(ok=True, wall=round(wall, 1),
                                      nodes={g[k]["class_type"] + ":" + k: round(v, 1) for k, v in timings.items()})
                    except Exception as exc:
                        record.update(ok=False, error=str(exc)[:400])
                record["peak_mib_card"] = peak.peak - max(hold, 0)
                print(json.dumps(record), flush=True)
                with open(out_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if not record["ok"]:
                    break
        finally:
            server.terminate()
            server.wait(60)
            if filler is not None:
                filler.stdin.close()
                filler.wait(30)
            time.sleep(3)


if __name__ == "__main__":
    main()
