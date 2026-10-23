"""The port against Tencent's DiT (captures/dit_oracle.pt from capture_dit.py): same int8 weights, same inputs.

Run on one GPU with ComfyUI on the path:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/AI/ComfyUI python dev/parity_dit.py
"""
import os
import sys
from pathlib import Path

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
LAB = Path(os.environ.get("HY_NOMAS_LAB", Path.home() / "AI/hy-nomas-lab"))
CKPT = os.environ.get("HY_NOMAS_DIT", str(Path.home() / "AI/hf_originals/Hy-Nomas/DiT/distilled_int8/"
                                       "diffusion_pytorch_model_ema_0_995_int8.safetensors"))

import comfy.model_management as mm  # noqa: E402
from hy_nomas import loader, world as W  # noqa: E402


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm()).item()


def main():
    matmul = sys.argv[1] if len(sys.argv) > 1 else "exact"
    ref = torch.load(LAB / "captures/dit_oracle.pt")
    inp = ref["inputs"]
    patcher = loader.load_dit(CKPT, matmul=matmul, disable_dynamic=True)
    mm.load_models_gpu([patcher])
    dm = patcher.model.diffusion_model
    dev = mm.get_torch_device()
    L0, L1 = inp["L0"], inp["L1"]
    x0, x1, clean0 = (inp[k].to(dev) for k in ("x0", "x1", "clean0"))
    txt0 = inp["txt0"][:, :L0].to(dev)
    txt1 = inp["txt1"][:, :L1].to(dev)
    t = inp["t_denoise"].to(dev)
    tc = inp["t_cache"].to(dev)
    results = {}
    with torch.no_grad():
        results["A"] = rel(dm._forward(x0, t, txt0, nomas={}), ref["A"])
        results["B"] = rel(dm._forward(x0, t, txt0, nomas={"action_offsets": inp["offs0"],
                                                          "action_windows": inp["wins0"]}), ref["B"])
        results["A_cam"] = rel(dm._forward(x0, t, txt0, nomas={"camera": inp["ext0"]}), ref["A_cam"])
        results["cam_tokens"] = rel(dm.camera_tokens(inp["ext0"].to(dev), torch.bfloat16), ref["cam0_tokens"])

        kv = {}
        dm._forward(clean0, tc, txt0, nomas={"action_offsets": inp["offs0"], "action_windows": inp["wins0"]},
                    capture=lambda i, k, v: kv.__setitem__(i, (k[0], v[0])))
        lv = clean0.shape[2] * clean0.shape[3] * clean0.shape[4]
        valid = lv + L0
        results["C_k"] = rel(torch.stack([kv[i][0] for i in range(len(kv))]).cpu(), ref["C_k"][:, :valid])
        results["C_v"] = rel(torch.stack([kv[i][1] for i in range(len(kv))]).cpu(), ref["C_v"][:, :valid])

        for name, int8 in (("D", False), ("D_int8", True)):
            record = W.make_record(0, 0, clean0.shape[2], clean0.shape[3] * clean0.shape[4], L0, inp["action_start0"],
                                   [kv[i] for i in range(len(kv))], kv_int8=int8)
            world = W.World(records=(record,), latents=(clean0.cpu(),), kv_int8=int8)
            out = dm._forward(x1, t, txt1, nomas={"history": world.history()})
            results[name] = rel(out, ref["D"])
            world.history().release()
    for k, v in results.items():
        print(f"{k:10s} rel err {v:.2e}")


if __name__ == "__main__":
    main()
