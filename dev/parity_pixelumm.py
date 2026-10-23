"""The ported PixelUMM decoder against Tencent's (captures/pixelumm_oracle.pt): dequant and fp8 matmul modes."""
import os
import sys
from pathlib import Path

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
LAB = Path(os.environ.get("HY_NOMAS_LAB", Path.home() / "AI/hy-nomas-lab"))
F8 = os.environ.get("HY_NOMAS_DECODER", str(Path.home() / "AI/hf_originals/Hy-Nomas/VAE/PixelUMM_FT_FP8/"
                                              "diffusion_pytorch_model_fp8.safetensors"))

from hy_nomas import pixelumm as P  # noqa: E402


def main():
    modes = sys.argv[1:] or ["dequant"]
    ref = torch.load(LAB / "captures/pixelumm_oracle.pt")
    state, _ = P.load_state_dict(F8)
    for mode in modes:
        with torch.device("meta"):
            dec = P.PixelUMMDecoder(ops=P.DecoderOps.with_matmul(mode))
        missing = dec.load_state_dict(dict(state), strict=False, assign=True)
        assert not missing.missing_keys and not missing.unexpected_keys, missing
        dec = dec.to("cuda").eval()
        for name, case in ref.items():
            with torch.no_grad():
                y = dec(case["latent"].cuda(), case["xt"].cuda()).float().cpu()
            r = case["x0"]
            print(f"{mode:8s} {name:6s} rel {((y - r).norm() / r.norm()).item():.2e} "
                  f"max {(y - r).abs().max().item():.3f}")


if __name__ == "__main__":
    main()
