#!/usr/bin/env bash
# A throwaway ComfyUI on the RTX 4090 only (nvidia-smi index 1), with only this pack, its own dirs and no DB lock.
#   PORT=8199 ATTN=sage dev/serve.sh [extra ComfyUI args]
export CUDA_DEVICE_ORDER=PCI_BUS_ID
LAB="${HY_NOMAS_LAB:-$HOME/AI/hy-nomas-lab}/server"
mkdir -p "$LAB"/{user,output,input,temp}
case "${ATTN:-sage}" in
  sage) attn=(--use-sage-attention) ;; pytorch) attn=(--use-pytorch-cross-attention) ;; flash) attn=(--use-flash-attention) ;;
esac
cd "${COMFYUI_DIR:-$HOME/AI/ComfyUI}"
exec .venv/bin/python main.py --cuda-device "${GPU:-1}" --port "${PORT:-8199}" "${attn[@]}" --disable-auto-launch \
  --disable-all-custom-nodes --whitelist-custom-nodes ComfyUI-HyNomas \
  --user-directory "$LAB/user" --output-directory "$LAB/output" --input-directory "$LAB/input" \
  --temp-directory "$LAB/temp" --database-url sqlite:///:memory: "$@"
