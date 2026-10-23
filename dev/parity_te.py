"""ComfyUI's Qwen2.5-VL-7B (type qwen_image) through text.py against the transformers 4.57.1 oracle (te_oracle.pt).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/AI/ComfyUI python dev/parity_te.py <te file>...
"""
import os
import sys
from pathlib import Path

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
LAB = Path(os.environ.get("HY_NOMAS_LAB", Path.home() / "AI/hy-nomas-lab"))

import comfy.sd  # noqa: E402
from hy_nomas import text  # noqa: E402


def main():
    oracle = torch.load(LAB / "captures/te_oracle.pt")
    fp32 = LAB / "captures/te_oracle_fp32.pt"
    golden = torch.load(fp32)["items"] if fp32.exists() else None
    if golden is not None:
        hf = [((o["states"].float() - g["states"]).norm() / g["states"].norm()).item()
              for o, g in zip(oracle["items"], golden)]
        print(f"transformers bf16 vs fp32: worst {max(hf):.2e}")
    for path in sys.argv[1:]:
        clip = comfy.sd.load_clip(ckpt_paths=[path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
        errs = []
        for item in oracle["items"]:
            states = text.encode_ids(clip, item["ids"])
            ref = item["states"].float()
            got = states[0].float().cpu()
            errs.append(((got - ref).norm() / ref.norm()).item())
            cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1).min().item()
            line = f"  {Path(path).name}: {got.shape[0]} tokens rel {errs[-1]:.2e} min token cosine {cos:.5f}"
            if golden is not None:
                g = golden[oracle["items"].index(item)]["states"]
                line += f" | vs fp32 {((got - g).norm() / g.norm()).item():.2e}"
            print(line)
        print(f"{Path(path).name}: worst {max(errs):.2e}")
        del clip


if __name__ == "__main__":
    main()
