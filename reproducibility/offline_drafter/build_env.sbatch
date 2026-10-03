#!/bin/bash
# Build the eagle-train environment (~/envs/eagle-train): eagle-train pins transformers 4.51.1 and torch 2.6,
# which the fastrl env (transformers 4.57, torch 2.8) cannot import (transformers.utils.LossKwargs is gone).
# Submit from the repo root: sbatch reproducibility/offline_drafter/build_env.sbatch
#SBATCH --job-name=eagle-env
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
[ -d "$REPO/eagle-train" ] || { echo "submit from the repo root (or set FASTRL_REPO)"; exit 1; }
ENV=${EAGLE_ENV:-$HOME/envs/eagle-train}
export UV_CACHE_DIR=/oscar/scratch/$USER/uv-cache
module load cuda/12.9.0-cinr  # nvcc on PATH for DeepSpeed's setup; no ops are prebuilt (optimizer = torch AdamW)

uv venv --python 3.10 "$ENV"
source "$ENV/bin/activate"
# eagle-train/requirements.txt, plus what its scripts import (hydra, safetensors, pyarrow)
uv pip install torch==2.6.0 transformers==4.51.1 accelerate datasets pandas pyarrow torchdata wandb hydra-core \
    safetensors tqdm numpy setuptools wheel ninja
# DeepSpeed's setup.py imports torch, so build it against the torch installed above
DS_BUILD_OPS=0 uv pip install --no-build-isolation deepspeed
# PyPI torch 2.6 wheels use the pre-C++11 ABI
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Import checks need a GPU node (DeepSpeed's Triton kernels look for a driver at import, and transformers
# imports DeepSpeed), so they run as a preflight in datagen.sbatch (check_env.py). Here: installed versions.
uv pip list | grep -E '^(torch|transformers|deepspeed|flash-attn|flash_attn|triton) '
