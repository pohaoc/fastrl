#!/bin/bash
# Stage 1: sample Qwen2.5-7B rollouts (gen_rollouts.py) on 1x L40S, plain decoding (no speculation), fa3.
# Submit from the repo root: DATASET=dapo|sql sbatch reproducibility/offline_drafter/gen.sbatch
#SBATCH --job-name=eagle-gen
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
source "$REPO/reproducibility/offline_drafter/paths.sh"
export HF_HUB_OFFLINE=1
echo "job=$SLURM_JOB_ID node=$(hostname) DATASET=$DATASET out=$ROLLOUTS commit=$(git -C "$REPO" rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
cd "$REPO"
python reproducibility/offline_drafter/gen_rollouts.py --dataset "$DATASET" --out "$ROLLOUTS" "$@"
