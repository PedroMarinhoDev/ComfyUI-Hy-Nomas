"""The fused kernels against the torch arithmetic they replace (model.py), on random bf16 data."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def test_adaln():
    from hy_nomas import fused, model as M
    x = (torch.randn(2, 2048, 777, device="cuda") * 3).bfloat16().transpose(1, 2)
    shift = torch.randn(2, 2048, device="cuda").bfloat16()
    scale = torch.randn(2, 2048, device="cuda").bfloat16()
    ref = M.modulate(M.layer_norm(x), shift, scale).to(torch.bfloat16)
    got = fused.adaln(x, shift, scale)
    mismatch = (got != ref).float().mean().item()
    assert mismatch < 1e-3, mismatch
    assert (got.float() - ref.float()).abs().max().item() < 0.05


def test_qk_norm_rope():
    from hy_nomas import fused, model as M
    b, l, h, d = 2, 96, 16, 128
    lin = (torch.randn(b, l, h * d, device="cuda") * 5).bfloat16()
    x = lin.view(b, l, h, d)
    norm = M.RMSNorm(d).cuda().bfloat16()
    with torch.no_grad():
        norm.weight.copy_(torch.rand(d) + 0.5)
    rope = M.rope_freqs([16, 56, 56], 256, 4, 4, 6, device="cuda")
    with torch.no_grad():
        ref = M.apply_rope(norm(x).to(torch.bfloat16), *rope)
        got = fused.qk_norm_rope(x, norm.weight, rope)
        ref_plain = norm(x).to(torch.bfloat16)
        got_plain = fused.qk_norm_rope(x, norm.weight, None)
    assert (got_plain != ref_plain).float().mean().item() < 1e-3
    assert (got != ref).float().mean().item() < 1e-2
    assert ((got.float() - ref.float()).norm() / ref.float().norm()).item() < 1e-3


def test_gated_residual():
    from hy_nomas import fused
    x = torch.randn(1, 2048, 500, device="cuda").bfloat16().transpose(1, 2)   # strided, like the patch embed
    y = torch.randn(1, 500, 2048, device="cuda").bfloat16()
    g = torch.randn(1, 2048, device="cuda").bfloat16()
    ref = x + y * g.unsqueeze(1)
    assert torch.equal(fused.gated_residual(x, y, g), ref)


def test_adaln_quant_matches_unfused():
    from hy_nomas import fused, kernels
    x = (torch.randn(1, 2048, 300, device="cuda") * 3).bfloat16().transpose(1, 2)
    shift = torch.randn(1, 2048, device="cuda").bfloat16()
    scale = torch.randn(1, 2048, device="cuda").bfloat16()
    q, s = kernels.adaln_quant(x, shift, scale)
    q_ref, s_ref = kernels.quantize_activations(fused.adaln(x, shift, scale).view(-1, 2048))
    assert (q != q_ref).float().mean().item() < 1e-3
    assert torch.allclose(s, s_ref, rtol=1e-3)


def test_gemm_gelu_quant_epilogue():
    from hy_nomas import kernels, ops
    torch.manual_seed(0)
    x = torch.randn(1000, 2048, device="cuda").bfloat16()
    w = torch.randn(8192, 2048, device="cuda") * 0.02
    bias = (torch.randn(8192, device="cuda") * 0.1).bfloat16()
    qw, sw = ops.quantize_g128(w)
    qa, sa = kernels.quantize_activations(x)
    plain = kernels.block_gemm(qa, sa, qw, sw, bias)
    gelu = kernels.block_gemm(qa, sa, qw, sw, bias, epilogue="gelu")
    ref = torch.nn.functional.gelu(plain, approximate="tanh")
    assert (gelu.float() - ref.float()).abs().max().item() <= 2 ** -6 * ref.float().abs().max().item()
    qh, sh = kernels.block_gemm(qa, sa, qw, sw, bias, epilogue="gelu_quant")
    qh_ref, sh_ref = kernels.quantize_activations(gelu)
    assert (qh != qh_ref).float().mean().item() < 1e-3
    assert torch.allclose(sh, sh_ref, rtol=1e-3)
