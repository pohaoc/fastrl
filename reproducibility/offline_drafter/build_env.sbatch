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

python -c "import torch, transformers, deepspeed, flash_attn; print('torch', torch.__version__, 'transformers', transformers.__version__, 'deepspeed', deepspeed.__version__, 'flash_attn', flash_attn.__version__)"

# The released drafter's tensors must map onto eagle-train's model (what --init_draft_path loads)
cd "$REPO/eagle-train"
HF_HOME=/oscar/scratch/$USER/hf HF_HUB_OFFLINE=1 python - <<'EOF'
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from model.qwen2_eagle import Qwen2ForCausalLMEagle
cfg = AutoConfig.from_pretrained(snapshot_download("Qwen/Qwen2.5-7B"))
cfg.num_hidden_layers = 1
with torch.device("meta"):
    m = Qwen2ForCausalLMEagle(cfg)
keys = {k: tuple(v.shape) for k, v in m.state_dict().items()}
sd = torch.load(snapshot_download("mit-han-lab/Qwen2.5-7B-Eagle-RL") + "/pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True)
ck = {(k if k.startswith(("model.", "lm_head")) else "model." + k): tuple(v.shape) for k, v in sd.items()}
print("model params:", len(keys), "checkpoint tensors:", len(ck))
print("missing from checkpoint:", sorted(set(keys) - set(ck)))
print("unexpected in checkpoint:", sorted(set(ck) - set(keys)))
bad = [k for k in set(keys) & set(ck) if keys[k] != ck[k]]
print("shape mismatches:", bad)
assert not (set(ck) - set(keys)) and not bad and set(keys) - set(ck) <= {"lm_head.weight"}
print("OK: Eagle-RL checkpoint maps onto eagle-train's Qwen2ForCausalLMEagle")
EOF
