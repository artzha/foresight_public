#!/bin/bash

uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install vllm==0.16.0
uv pip install qwen-vl-utils

uv pip install scipy matplotlib joblib hickle pydantic \
    opencv-python pillow hydra-core omegaconf pandas wandb

# grounding policy deps
uv pip install efficientnet_pytorch
uv pip install "depth-anything-v2 @ git+https://github.com/debOliveira/depth-anything-V2.git@7885bbc0647bc64d55ff5803561ea2c7dea1af72"

# Optional: only needed to serve the reward model role (vlm_reward.enabled: true).
# uv pip install "verl @ git+https://github.com/volcengine/verl.git"
