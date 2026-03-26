#!/usr/bin/env bash
set -euo pipefail

conda create -y --name encoders --clone encoders_heads
conda run -n encoders python -m pip install --upgrade pip setuptools wheel
conda run -n encoders python -m pip install \
  "transformers>=4.56.0" "timm>=1.0.20" "huggingface_hub>=0.24.0" \
  accelerate peft safetensors pillow numpy pandas pyarrow tqdm pyyaml h5py \
  einops sentencepiece scipy scikit-image opencv-python ftfy regex \
  tensorboardX torchsummary numba

# InternViT is loaded via local patched remote code with flash-attn disabled.
# No flash-attn installation is required for this repository workflow.
