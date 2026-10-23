"""Load a Hy-Nomas DiT checkpoint into a native ComfyUI MODEL.

Accepted files (detected from the tensors, not the name):
* Tencent's INT8 file (`nomas-int8-bf16-single-v1`): `weight_q` int8 + `weight_scale` fp32 per Linear, everything
  else bf16. Loaded as is, each pair one `NomasInt8G128Layout` weight (`ops.py`).
* Tencent's fp32 / any bf16 file with plain `weight` tensors (distilled, base AR). Cast to bf16, or quantized to
  the same int8 layout at load when asked.

The model is built the way the HunyuanImage-3.0 pack builds its own: a `BaseModel` subclass, the patcher class
looked up at call time (main.py swaps `CoreModelPatcher` for the dynamic one), `assign=` loading so dynamic VRAM
can stream the weights from the file's pages, and `cached_patcher_init` so Select Model Device can rebuild it.
"""
import json
import logging
import os

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.utils

from . import ops as nomas_ops
from .model_base import HyNomasModel, HyNomasModelConfig

UNET_CONFIG = {
    "in_channels": 32, "out_channels": 32, "hidden_size": 2048, "heads_num": 16, "mlp_width_ratio": 4,
    "mm_double_blocks_depth": 54, "rope_dim_list": [16, 56, 56], "rope_theta": 256, "text_states_dim": 3584,
}
# small modules the int8 path keeps unquantized when quantizing a float checkpoint, as Tencent's file does
KEEP_FLOAT = ("action_modules.", "extrinsics_embedding_camctrl", "c2ws_hidden_states_layer", "img_in.")


def detect(state_dict, metadata):
    fmt = (metadata or {}).get("format")
    if fmt == "nomas-int8-bf16-single-v1" or any(k.endswith(".weight_q") for k in state_dict):
        return "int8"
    if "double_blocks.0.img_attn_q.weight" in state_dict:
        return "float"
    raise ValueError("not a Hy-Nomas DiT checkpoint (no double_blocks.0.img_attn_q weights)")


def config_from_state_dict(state_dict):
    cfg = dict(UNET_CONFIG)
    depth = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("double_blocks."))
    cfg["mm_double_blocks_depth"] = depth
    cfg["camera"] = any(k.startswith("action_modules.") for k in state_dict)
    return cfg


def quantize_state_dict(state_dict):
    """A float checkpoint's Linear weights -> Tencent's int8 g128 pairs (the reference's `quantize_weight`)."""
    out = {}
    for key, value in state_dict.items():
        if (key.endswith(".weight") and value.ndim == 2 and not key.startswith(KEEP_FLOAT)
                and value.shape[1] % 128 == 0):
            q, s = nomas_ops.quantize_g128(value)
            out[key[:-len("weight")] + "weight_q"] = q
            out[key[:-len("weight")] + "weight_scale"] = s
        else:
            out[key] = value
    return out


def _factory(path, matmul="exact", quantize=False, disable_dynamic=False):
    return load_dit(path, matmul=matmul, quantize=quantize, disable_dynamic=disable_dynamic)


def load_dit(path, matmul="exact", quantize=False, disable_dynamic=False):
    state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    kind = detect(state_dict, metadata)
    if kind == "float":
        state_dict = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in state_dict.items()}
        if quantize:
            state_dict = quantize_state_dict(state_dict)
            kind = "int8"
    unet_config = config_from_state_dict(state_dict)

    dtype = torch.bfloat16
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    operations = nomas_ops.NomasOps.with_matmul(matmul)
    model_config = HyNomasModelConfig(unet_config, dtype, operations)
    model = HyNomasModel(model_config, device=offload_device)

    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(model, load_device=load_device, offload_device=offload_device)
    patcher.cached_patcher_init = (_factory, (path, matmul, quantize))

    loaded = model.diffusion_model.load_state_dict(state_dict, strict=False, assign=patcher.is_dynamic())
    unexpected = [k for k in loaded.unexpected_keys]
    missing = [k for k in loaded.missing_keys]
    del state_dict
    if unexpected or missing:
        raise ValueError(f"Hy-Nomas checkpoint mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
                         f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})")
    logging.info("Hy-Nomas: loaded %s (%s weights, %d blocks, camera %s, matmul %s)", os.path.basename(path), kind,
                 unet_config["mm_double_blocks_depth"], unet_config["camera"], matmul)
    return patcher


def info(metadata):
    if not metadata:
        return {}
    keep = ("format", "model_type", "quantization", "group_size", "source_sha256")
    return {k: metadata[k] for k in keep if k in metadata}


def describe_file(path):
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as handle:
            return json.dumps(info(handle.metadata()))
    except Exception:
        return "{}"
