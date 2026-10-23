"""DCAE-V encoder parity: the port (weights from the PixelUMM file) vs `capture_dcae.py` (Tencent's DCAEV.encode,
weights from the DCAE-V .pt) on a still clip, a moving clip and multi-tile lengths. Expect bf16 conv-algorithm
level differences (~1e-3)."""
import os
import sys
from pathlib import Path

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
LAB = Path(os.environ.get("HY_NOMAS_LAB", Path.home() / "AI/hy-nomas-lab"))
WEIGHTS = os.environ.get("HY_NOMAS_ENCODER", str(Path.home() / "AI/hf_originals/Hy-Nomas/VAE/PixelUMM_FT_FP8/"
                                                 "diffusion_pytorch_model_fp8.safetensors"))

from hy_nomas import dcae  # noqa: E402


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    ref = torch.load(LAB / "captures/dcae_oracle.pt")
    patcher = dcae.load_encoder(WEIGHTS, disable_dynamic=True)
    worst = 0.0
    for name, case in ref.items():
        z = dcae.encode(patcher, case["video"].float())
        e = rel(z, case["latent"])
        worst = max(worst, e)
        print(f"{name:8s} {tuple(z.shape)}  rel {e:.2e}  max abs {(z - case['latent']).abs().max().item():.3f}")
    print(f"worst {worst:.2e}")


if __name__ == "__main__":
    main()
