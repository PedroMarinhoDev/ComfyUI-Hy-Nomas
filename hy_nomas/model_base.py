"""Hy-Nomas as a native ComfyUI `BaseModel`, driven by the stock samplers one segment at a time.

The sampling contract (`inference.yaml`): flow matching, `shift` 5, 8 Euler steps, guidance 1. In ComfyUI that is
`ModelType.FLOW` (`ModelSamplingDiscreteFlow` + `CONST`), sampler `euler`, scheduler `simple`, cfg 1. `simple`
picks sigma(1 - i/8) from the 1000-entry table, which is the reference's shifted `linspace(1, 0, 9)`.

The conditioning carries one `NomasCond` per prompt (text layout, history, camera). It never batches with a
different one: a CFG negative runs as its own forward, like the reference's separate guidance branch.
"""
import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base


class HyNomasLatent(comfy.latent_formats.LatentFormat):
    """DCAE-V f32t4c32: 32 channels, /32 in space, /4 in time. The DiT works on the raw latents
    (`PixelUMMDCAELatentDecoder.scaling_factor` 1.0, no shift)."""
    latent_channels = 32
    latent_dimensions = 3
    spacial_downscale_ratio = 32
    temporal_downscale_ratio = 4
    scale_factor = 1.0


class NomasCond(comfy.conds.CONDConstant):
    """A dict of per-prompt inputs; equal only to itself, so two prompts never share a forward."""

    def can_concat(self, other):
        return self.cond is other.cond

    def concat(self, others):
        return self.cond


class HyNomasModelConfig:
    def __init__(self, unet_config, dtype, operations, quant_config=None):
        self.unet_config = dict(unet_config, dtype=dtype)
        self.latent_format = HyNomasLatent()
        self.sampling_settings = {"shift": 5.0, "multiplier": 1000}
        self.quant_config = quant_config
        self.manual_cast_dtype = None
        self.custom_operations = operations
        self.optimizations = {}
        self.model_type = None
        self.memory_usage_factor = 1.6
        self.supported_inference_dtypes = [torch.bfloat16]


class HyNomasModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        from .model import HyNomasDiT
        super().__init__(model_config, comfy.model_base.ModelType.FLOW, device=device, unet_model=HyNomasDiT)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        nomas = kwargs.get("nomas", None)
        if nomas is not None:
            out["nomas"] = NomasCond(nomas)
        return out

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        # the history is staged on the GPU once per sampling run and dropped when the run's sigmas go away
        nomas = kwargs.get("nomas", None)
        if isinstance(nomas, dict) and nomas.get("world") is not None:
            sigmas = transformer_options.get("sample_sigmas", None)
            nomas = dict(nomas, history=nomas["world"].history(), run_key=sigmas)
            kwargs["nomas"] = nomas
        return super()._apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def memory_required(self, input_shape, cond_shapes={}):
        # activations of one block over the segment's tokens (the history is staged separately)
        tokens = input_shape[0] * input_shape[2] * input_shape[3] * input_shape[4]
        return tokens * 2048 * 2 * 14 + 512 * 1024 * 1024
