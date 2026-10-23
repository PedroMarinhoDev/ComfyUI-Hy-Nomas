"""Tencent's INT8 DiT weights as a ComfyUI quantized layout.

`DiT/distilled_int8/diffusion_pytorch_model_ema_0_995_int8.safetensors` stores every Linear as
`<name>.weight_q` (int8 [N, K]) and `<name>.weight_scale` (fp32 [N, ceil(K / 128)]): symmetric absmax per
output row and per 128-wide input group (`int8_stored_bf16_linear.quantize_weight`). The reference dequantizes
the whole weight to bf16 on every call and runs a bf16 GEMM.

Here the pair becomes one `QuantizedTensor` of `NomasInt8G128Layout`, so ComfyUI's model management sees one
weight per Linear: dynamic VRAM streams it, `.to()` moves both tensors, LoRA patches dequantize it like any other
layout. The Linear below picks the matmul:

* `exact`: dequantize to bf16, `F.linear` - the reference's arithmetic, bit for bit.
* `int8`: W8A8 on the int8 tensor cores (`kernels.int8_g128_linear`), activations quantized per token and per
  128-group so the weight groups stay intact. Much faster on Ada/Ampere; not bit-exact.
"""
import contextlib
import dataclasses
import json
import logging
import os

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.quant_ops import QuantizedTensor

try:
    from comfy_kitchen.tensor import BaseLayoutParams, QuantizedLayout, register_layout_class
    _CK = True
except ImportError:  # comfy-kitchen missing: the exact path still works through plain tensors
    _CK = False

GROUP = 128
LAYOUT = "NomasInt8G128Layout"
# "exact" | "int8"; the loader node sets it per model, the env var overrides for benchmarks
MATMUL_ENV = "HY_NOMAS_MATMUL"


def quantize_g128(weight):
    """Tencent's `quantize_weight`: symmetric per-row, per-128-column-group int8, half away from zero."""
    n, k = weight.shape
    groups = (k + GROUP - 1) // GROUP
    padded = F.pad(weight.float(), (0, groups * GROUP - k))
    rows = padded.reshape(n, groups, GROUP)
    peak = rows.abs().amax(dim=-1)
    scale = torch.where(peak > 0, peak / 127.0, 1.0)
    normalized = rows / scale.unsqueeze(-1)
    rounded = (normalized.abs() + 0.5).floor() * normalized.sign()
    q = rounded.clamp_(-127, 127).to(torch.int8).reshape(n, -1)[:, :k].contiguous()
    return q, scale.contiguous()


def dequantize_g128(qdata, scale, dtype=torch.bfloat16):
    """The reference's `dequant_weight`: fp32 product, one cast to `dtype`."""
    n, k = qdata.shape
    if k % GROUP == 0:
        weight = (qdata.view(n, k // GROUP, GROUP).float() * scale.unsqueeze(-1)).view(n, k)
    else:
        weight = qdata.float() * scale.repeat_interleave(GROUP, dim=1)[:, :k]
    return weight.to(dtype)


if _CK:
    class NomasInt8G128Layout(QuantizedLayout):
        MIN_SM_VERSION = None

        @dataclasses.dataclass(frozen=True)
        class Params(BaseLayoutParams):
            pass

        @classmethod
        def quantize(cls, tensor, scale=None, **kwargs):
            q, s = quantize_g128(tensor)
            return q, cls.Params(scale=s, orig_dtype=tensor.dtype, orig_shape=tuple(tensor.shape))

        @classmethod
        def dequantize(cls, qdata, params):
            return dequantize_g128(qdata, params.scale, params.orig_dtype)

        @classmethod
        def get_plain_tensors(cls, qtensor):
            return qtensor._qdata, qtensor._params.scale

        @classmethod
        def state_dict_tensors(cls, qdata, params):
            return {"_q": qdata, "_scale": params.scale}

    register_layout_class(LAYOUT, NomasInt8G128Layout)


def make_weight(qdata, scale, dtype=torch.bfloat16):
    if not _CK:
        raise RuntimeError("Hy-Nomas INT8 weights need comfy-kitchen (it ships with ComfyUI)")
    params = NomasInt8G128Layout.Params(scale=scale, orig_dtype=dtype, orig_shape=tuple(qdata.shape))
    return QuantizedTensor(qdata, LAYOUT, params)


def matmul_mode(default="exact"):
    return os.environ.get(MATMUL_ENV, default)


class NomasOps(comfy.ops.manual_cast):
    """ComfyUI's cast-on-use ops, with a Linear that also loads Tencent's `weight_q`/`weight_scale` pairs.
    `with_matmul` makes the ops class a loader hands the model."""

    @classmethod
    def with_matmul(cls, matmul):
        linear = type("Linear", (cls.Linear,), {"default_matmul": matmul})
        return type(f"NomasOps_{matmul}", (cls,), {"Linear": linear})

    class Linear(comfy.ops.manual_cast.Linear):
        default_matmul = "exact"

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self._has_bias = bias
            self._dtype = dtype
            self.matmul = None   # None: the ops class's default

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                  error_msgs):
            q = state_dict.pop(prefix + "weight_q", None)
            scale = state_dict.pop(prefix + "weight_scale", None)
            plain = state_dict.pop(prefix + "weight", None)
            bias = state_dict.pop(prefix + "bias", None)
            if q is not None:
                if scale is None:
                    error_msgs.append(f"{prefix}weight_q without {prefix}weight_scale")
                    return
                if tuple(q.shape) != (self.out_features, self.in_features):
                    error_msgs.append(f"{prefix}weight_q has shape {tuple(q.shape)}, expected "
                                      f"{(self.out_features, self.in_features)}")
                    return
                weight = make_weight(q, scale.float(), self._dtype or torch.bfloat16)
            elif plain is not None:
                weight = plain if self._dtype is None else plain.to(self._dtype)
            else:
                missing_keys.append(prefix + "weight")
                return
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            if bias is not None:
                self.bias = torch.nn.Parameter(bias if self._dtype is None else bias.to(self._dtype),
                                               requires_grad=False)
            elif self._has_bias:
                missing_keys.append(prefix + "bias")

        def convert_weight(self, weight, inplace=False, **kwargs):
            return weight.dequantize() if isinstance(weight, QuantizedTensor) else weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            if isinstance(self.weight, QuantizedTensor):
                q, s = quantize_g128(weight)
                weight = make_weight(q, s, self.weight._params.orig_dtype)
            if return_weight:
                return weight
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def forward_comfy_cast_weights(self, input):
            with comfy.ops.CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                if isinstance(weight, QuantizedTensor):
                    mode = self.matmul or matmul_mode(self.default_matmul)
                    qdata, scale = weight._qdata, weight._params.scale
                    if mode == "int8" and input.is_cuda:
                        from .kernels import int8_g128_linear
                        return int8_g128_linear(input, qdata, scale, bias)
                    weight = dequantize_g128(qdata, scale, input.dtype)
                return F.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            return self.forward_comfy_cast_weights(*args, **kwargs)

        def fast_int8(self, input):
            """Whether this linear can take pre-quantized int8 activations (`int8_weights`)."""
            return (isinstance(self.weight, QuantizedTensor) and input.is_cuda
                    and (self.matmul or matmul_mode(self.default_matmul)) == "int8"
                    and not self.weight_function and not self.bias_function)

        @contextlib.contextmanager
        def int8_weights(self, input):
            """(int8 codes, scales, bias) on the input's device, cast the way the rest of ComfyUI casts."""
            with comfy.ops.CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                yield weight._qdata, weight._params.scale, bias


def quantized_linears(module):
    return [m for m in module.modules() if isinstance(m, NomasOps.Linear) and isinstance(m.weight, QuantizedTensor)]


def describe(module):
    linears = quantized_linears(module)
    size = sum(m.weight._qdata.numel() for m in linears)
    logging.info("Hy-Nomas: %d int8 g128 linears (%.2f G weights)", len(linears), size / 1e9)
    return json.dumps({"int8_linears": len(linears)})
