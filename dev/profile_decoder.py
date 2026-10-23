"""One PixelUMM decoder window (10 latents) at a given size, warmed up, with a kernel breakdown.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/AI/ComfyUI python dev/profile_decoder.py --matmul fp8
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
F8 = os.environ.get("HY_NOMAS_DECODER", str(Path.home() / "AI/hf_originals/Hy-Nomas/VAE/PixelUMM_FT_FP8/"
                                              "diffusion_pytorch_model_fp8.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matmul", default="fp8")
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    from hy_nomas import pixelumm as P
    state, _ = P.load_state_dict(F8)
    with torch.device("meta"):
        dec = P.PixelUMMDecoder(ops=P.DecoderOps.with_matmul(args.matmul))
    dec.load_state_dict(state, strict=False, assign=True)
    dec = dec.to("cuda").eval()
    h, w = args.height // 32, args.width // 32
    lat = torch.randn(1, 32, args.frames, h, w, device="cuda", dtype=torch.bfloat16)
    xt = torch.randn(1, 3, 4 * args.frames, h * 32, w * 32, device="cuda")
    with torch.no_grad():
        for _ in range(2):
            dec(lat, xt)
        torch.cuda.synchronize()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            dec(lat, xt)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        print(f"window: {min(times):.3f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if args.profile:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                dec(lat, xt)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=60))


if __name__ == "__main__":
    main()
