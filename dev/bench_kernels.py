"""Block-scaled W8A8 kernels: error against the dequantized bf16 linear, and speed against it."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy_nomas import kernels, ops, pixelumm  # noqa: E402


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def main():
    torch.manual_seed(0)
    dev = "cuda"
    for m, k, n in ((26400, 2048, 2048), (26400, 2048, 8192), (26400, 8192, 2048), (35200, 4096, 12288),
                    (35200, 12288, 4096), (300, 2048, 12288)):
        x = (torch.randn(m, k, device=dev) * 2).to(torch.bfloat16)
        w = torch.randn(n, k, device=dev) * 0.02
        bias = torch.randn(n, device=dev).to(torch.bfloat16)
        q, s = ops.quantize_g128(w)
        wd = ops.dequantize_g128(q, s)
        ref = torch.nn.functional.linear(x, wd, bias)
        got = kernels.int8_g128_linear(x, q, s, bias)
        err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        t_bf16 = timeit(lambda: torch.nn.functional.linear(x, wd, bias))
        t_deq = timeit(lambda: torch.nn.functional.linear(x, ops.dequantize_g128(q, s), bias))
        t_int8 = timeit(lambda: kernels.int8_g128_linear(x, q, s, bias))
        line = (f"M{m:6d} K{k:5d} N{n:5d}  int8 err {err:.2e}  ms: bf16 {t_bf16:.2f}  dequant+bf16 {t_deq:.2f}  "
                f"int8 {t_int8:.2f} ({t_deq / t_int8:.2f}x)")
        if k % 128 == 0 and n % 128 == 0:
            qf, sf = pixelumm.NomasFP8Block128Layout.quantize(w)
            sf = sf.scale
            wf = pixelumm.dequantize_fp8_block(qf, sf)
            ref8 = torch.nn.functional.linear(x, wf, bias)
            got8 = kernels.block_linear(x, qf, sf.repeat_interleave(128, 0), bias, kind="fp8")
            err8 = ((got8.float() - ref8.float()).norm() / ref8.float().norm()).item()
            t_fp8 = timeit(lambda: kernels.block_linear(x, qf, sf.repeat_interleave(128, 0), bias, kind="fp8"))
            line += f"  | fp8 err {err8:.2e} {t_fp8:.2f} ms"
        print(line, flush=True)


if __name__ == "__main__":
    main()
