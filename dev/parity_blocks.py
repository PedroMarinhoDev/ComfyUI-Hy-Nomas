"""Teacher-forced parity: each captured reference block's exact inputs through the port's block.

Whole-model outputs on random inputs are chaotic (a 1e-3 input perturbation moves the output ~20%), so the
port is checked piece by piece: refiner, timestep embedder, RoPE, double blocks (plain, action mask, history),
camera modules and the final layer. Expect bf16-level differences (~1e-3 to 1e-2).
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
from hy_nomas import loader, model as M  # noqa: E402


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main():
    matmul = sys.argv[1] if len(sys.argv) > 1 else "exact"
    ref = torch.load(LAB / "captures/dit_blocks_oracle.pt")
    inp = ref["inputs"]
    patcher = loader.load_dit(CKPT, matmul=matmul, disable_dynamic=True)
    mm.load_models_gpu([patcher])
    dm = patcher.model.diffusion_model
    dev = mm.get_torch_device()
    bf = torch.bfloat16
    worst = 0.0
    with torch.no_grad():
        for case, txt_key, valid in (("B", "txt0", inp["L0"]), ("A_cam", "txt0", inp["L0"]), ("D", "txt1", inp["L1"])):
            rec = ref[case]
            t = inp["t"].to(dev)
            line = [case]
            vec = dm.time_in(t, bf)
            line.append(f"time_in {rel(vec, rec['time_in.out']):.1e}")
            txt = dm.txt_in(inp[txt_key][:, :valid].to(dev), t, bf, {})
            line.append(f"txt_in {rel(txt, rec['txt_in.out'][:, :valid]):.1e}")
            print("  ".join(line))
            x = inp["x1" if case == "D" else "x0"]
            tt, h, w = x.shape[2:]
            offset = 3 if case == "D" else 0
            rope = M.rope_freqs(dm.rope_dim_list, dm.rope_theta, tt, h, w, t_offset=offset, device=dev)
            for i in inp["blocks"]:
                bin_ = rec[f"b{i}.in"]
                rcos, rsin = rec[f"b{i}.freqs"]
                img = bin_["img"].to(dev, bf)
                txt_i = bin_["txt"][:, :valid].to(dev, bf)
                vec_i = bin_["vec"].to(dev, bf)
                history = None
                if case == "D":
                    vis = ref["D_mask"]
                    history = (ref["C_k"][i][vis].unsqueeze(0).to(dev, bf), ref["C_v"][i][vis].unsqueeze(0).to(dev, bf))
                plan = M.AttentionPlan(img.shape[1], valid,
                                       inp["offs0"] if case == "B" else None, inp["wins0"] if case == "B" else None,
                                       h * w, dev)
                out_img, out_txt = dm.double_blocks[i](img, txt_i, vec_i, rope, history, plan, {})
                r_img, r_txt = rec[f"b{i}.out"]
                e_img, e_txt = rel(out_img, r_img), rel(out_txt, r_txt[:, :valid])
                msg = f"  block {i:2d} rope {rel(rope[0], rcos):.0e}/{rel(rope[1], rsin):.0e} img {e_img:.1e} txt {e_txt:.1e}"
                worst = max(worst, e_img, e_txt)
                if case == "A_cam":
                    cam = dm.camera_tokens(inp["ext0"].to(dev), bf)
                    frame_out = dm.action_modules[i].frame_output(cam)
                    b = r_img.shape[0]
                    cam_img = (r_img.to(dev, bf).view(b, tt, h * w, -1) + frame_out.unsqueeze(2)).view(b, tt * h * w, -1)
                    e_cam = rel(cam_img, rec[f"cam{i}.out"])
                    worst = max(worst, e_cam)
                    msg += f" camera {e_cam:.1e}"
                print(msg)
            fx, fvec = rec["final.args"][:2]
            out = dm.final_layer(fx.to(dev, bf), fvec.to(dev, bf))
            e = rel(out, rec["final.out"])
            worst = max(worst, e)
            print(f"  final {e:.1e}")
    print(f"worst {worst:.2e}")


if __name__ == "__main__":
    main()
