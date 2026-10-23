"""One denoising step of the DiT at a given size, warmed up, with a kernel-level breakdown (torch.profiler).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/AI/ComfyUI python dev/profile_step.py \
        --frames 24 --height 704 --width 1280 --history 18 --matmul int8 --attention sage
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
CKPT = os.environ.get("HY_NOMAS_DIT", str(Path.home() / "AI/hf_originals/Hy-Nomas/DiT/distilled_int8/"
                                       "diffusion_pytorch_model_ema_0_995_int8.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--history", type=int, default=0, help="history latent frames (fake, int8)")
    ap.add_argument("--text", type=int, default=240)
    ap.add_argument("--matmul", default="int8")
    ap.add_argument("--attention", default="sage")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    import comfy.cli_args
    comfy.cli_args.args.use_sage_attention = args.attention == "sage"
    comfy.cli_args.args.use_pytorch_cross_attention = args.attention == "pytorch"
    comfy.cli_args.args.use_flash_attention = args.attention == "flash"
    import comfy.model_management as mm
    from hy_nomas import loader, world as W

    patcher = loader.load_dit(CKPT, matmul=args.matmul, disable_dynamic=True)
    mm.load_models_gpu([patcher])
    dm = patcher.model.diffusion_model
    dev = mm.get_torch_device()
    h, w = args.height // 32, args.width // 32
    x = torch.randn(1, 32, args.frames, h, w, device=dev, dtype=torch.bfloat16)
    ctx = torch.randn(1, args.text, 3584, device=dev, dtype=torch.bfloat16) * 3
    t = torch.tensor([937.5], device=dev)
    nomas = {}
    if args.history:
        tpf = h * w
        kv = []
        for _ in range(len(dm.double_blocks)):
            k = torch.randn(args.history * tpf + 200, 16, 128, device=dev, dtype=torch.bfloat16)
            kv.append((k, k))
        record = W.make_record(0, 0, args.history, tpf, 200, 200, kv, kv_int8=True)
        world = W.World(records=(record,), latents=(torch.zeros(1, 32, args.history, h, w),))
        nomas["history"] = world.history()
    print(f"tokens: video {args.frames * h * w}, text {args.text}, history "
          f"{len(nomas['history']) if 'history' in nomas else 0}", flush=True)
    with torch.no_grad():
        for _ in range(2):
            dm._forward(x, t, ctx, nomas=nomas)
        torch.cuda.synchronize()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            dm._forward(x, t, ctx, nomas=nomas)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        print(f"step: {min(times):.3f}s (min of 3), peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
              flush=True)
        if args.profile:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                dm._forward(x, t, ctx, nomas=nomas)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))


if __name__ == "__main__":
    main()
