# Hy-Nomas port: spec, status, how to test

Reference: `github.com/Tencent-Hunyuan/Hy-Nomas` @ 18974d4 (cloned at `~/AI/research/hy-nomas-src`), weights from
`huggingface.co/tencent/Hy-Nomas` @ dea9480 (`~/AI/hf_originals/Hy-Nomas`). Preset ported:
`configs/experiment/single_gpu_low_mem_inference.yaml` (kept in `docs/upstream/`).

## What the reference does (file:line in the reference repo)

**Prompts** (`hymm/inference.py:634` `_build_whole_item_prompts`, `:1133` `_load_from_json`): one JSON world ->
one JSON string per segment (`keep_keywords`, global fields only in segment 0, `visual_description_long` only in
segment 0, `description_change` dropped from segment 0, `contained_actions` moved last with times relative to the
segment). Segment boundaries in video frames = `int(seconds * 24)`; latent frames `(frames - 1) // 4 + 1`,
boundary `b // 4` (`hymm/pipeline/causal_ar.py:1728`). Ported in `hy_nomas/prompt.py`.

**Text** (`hymm/models/text_encoder/encoder.py`, `configs/constants.py`): Qwen2.5-VL-7B-Instruct language model,
chat template with the video system prompt (note the 8-space runs from backslash continuations), add_generation_prompt,
`hidden_states[-3]` (layer 25 output, no final norm), cropped after `<|im_start|>user\n` (crop_start 108), max 1280
tokens, padding masked everywhere. ComfyUI's `qwen_image` CLIP crops at the same marker; `hy_nomas/text.py` feeds it
token ids with `clip_layer(-3)`.

**Action masks** (`hymm/pipeline/base.py:429` `_compute_action_start`, `:573` offsets, `:684` windows;
`hymm/models/dit/nomas_cpt.py:105` `_build_intra_seg_cache_ranges`): token offsets by common prefixes of
action-truncated JSON variants; within a segment each action's video window sees head text + its own action text;
other segments see only the head (`cross_segment_text_head_only`). `hy_nomas/model.py` `AttentionPlan`.

**DiT** (`hymm/models/dit/nomas_cpt.py:887` block, `:3263` `forward_with_cache`): 54 dual-stream blocks, hidden
2048, 16x128 heads, qkv bias, QK RMSNorm, 3-D RoPE [16,56,56] theta 256 (interleaved pairs) on video only, token
refiner (2 blocks, SiLU MLP, masked mean context), final adaLN layer. Runs under `torch.autocast(bf16)`: LayerNorm,
modulation and `sum` in fp32. INT8 file: per-row, per-128-column absmax, half away from zero
(`int8_stored_bf16_linear.py`).

**Camera** (`hymm/models/camera/action_module.py`, `nomas_cpt.py:1931`): `extrinsics_attn` with
`per_chunk_camera_attn`, chunk 1 -> each latent frame's video attends to its single keyboard token: output =
`proj_keyboard(v)`. Poses: demo recording -> `demo_precomputed.prepare_turn` -> rotation gain 2.5
(`rotation_scale.py`) -> 3 front copies of the first pose -> 4 poses per token. Ported in `hy_nomas/camera.py`.

**Rollout** (`hymm/pipeline/causal_ar.py:1460`): per segment: Euler, 8 steps, shift 5, cfg 1, from noise; then the
cache write: `(1 - 0.206897) x0 + 0.206897 eps` at t = 206.897, same text/camera/masks, keys/values of
[video | text] kept per block. History for the next segment: `temporal_context` selection of 18 latent frames
(gamma 2) from the CPU archive, plus the text heads of segments with a selected frame
(`hymm/models/dit/modules/uniform_history_cache.py`); int8 KV, per token/head absmax. `hy_nomas/world.py`.

**PixelUMM** (`hymm/models/vae/pixelumm_decoder/`): 36-layer Qwen3-8B-shaped decoder, 1 step from per-latent-frame
seeded noise (seed 42), windows of 10 latents / overlap 5, cosine blend, local 8x8-token tile attention (3x3
neighbourhood) via flex attention. FP8 file: 128x128 block scales. `hy_nomas/pixelumm.py`.

**DCAE-V encoder** (`hymm/models/vae/autoencoder_dcae_v.py`, `dc_ae_v_f32t4c32_config`): only used to start a world
from an image (not part of Tencent's text-to-video flow). Chunk-causal 3x3x3 convs (chunks of 40 frames / 10 latents,
left padding from the previous chunk), RMSNorm3d, pixel-unshuffle downsampling with channel-averaging shortcuts,
40-frame encode tiles; latents go to the DiT unscaled. The weights are the `encoder.*` tensors of the PixelUMM file
(identical to `VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt`). `hy_nomas/dcae.py`.

**Start from an image** (this pack's addition): the picture held still for 3 latent frames (12 video frames) is
encoded, then added to the world by the ordinary cache pass under segment 0's text without its actions; generation
continues with segment 1, which sees the image as history. Latents along a still clip differ by 50-70% from each
other, so the clip is encoded, not one latent repeated. `session.start_world`.

## Verified

| What | How | Result |
|---|---|---|
| Prompt strings, token ids, crop, action offsets/windows | `tests/test_prompt_oracle.py` vs `capture_prompts.py` | exact |
| Text encoder | `dev/parity_te.py` vs transformers 4.57.1 bf16 + fp32 | ComfyUI bf16 weights vs fp32: 1e-6 (HF bf16 itself is 1-2% off fp32); fp8_scaled file 4-5% |
| DiT blocks, teacher-forced | `dev/parity_blocks.py` vs `capture_dit_blocks.py` | 1-4e-3 per block (attention kernel); final layer, time embedder, RoPE, camera exact |
| Cache K/V, history path, int8 KV | `dev/parity_dit.py` | consistent with the chaotic whole-model behaviour on random inputs (1e-3 input noise -> 20% output) |
| PixelUMM decoder | `dev/parity_pixelumm.py` vs `capture_pixelumm.py` | bit-exact (dequant mode), incl. local attention |
| DCAE-V encoder | `dev/parity_dcae.py` vs `capture_dcae.py` (still, moving, 44- and 80-frame tiles) | 4e-3 relative (bf16 convs) |
| Camera conversion | `dev/parity_camera.py` (ref-venv) | exact |
| Fused kernels | `tests/test_fused.py` | rounding-exact except rare 1-ulp |

Oracles live in `~/AI/hy-nomas-lab/captures/`; the capture scripts in `~/AI/hy-nomas-lab/` run in its `ref-venv`
(transformers 4.57.1, diffusers 0.36 over ComfyUI's torch).

## Testing

* Unit/oracle tests: `PYTHONPATH=~/AI/ComfyUI ~/AI/ComfyUI/.venv/bin/python -m pytest -q` (GPU tests need CUDA).
* Server: `dev/serve.sh` (4090 only, port 8199, only this pack, own dirs). `dev/run_world.py prompt.json
  --width 1280 --height 704` queues the per-segment graph and prints per-node times.
* Smaller cards: `dev/bench_lowvram.py --sizes 16,12`.
* One step, profiled: `dev/profile_step.py --frames 24 --matmul int8 --profile`.
