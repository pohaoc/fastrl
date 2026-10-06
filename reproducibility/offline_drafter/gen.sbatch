#!/bin/bash
# Stage 1: sample the target's own rollouts (gen_rollouts.py; target = $TARGET_MODEL from paths.sh) on 1x L40S,
# plain decoding (no speculation), fa3.
# Submit from the repo root: DATASET=dapo|sql sbatch reproducibility/offline_drafter/gen.sbatch
#SBATCH --job-name=eagle-gen
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu3001                 # uncorrectable ECC errors (2026-09-30, 2026-10-04)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
source "$REPO/reproducibility/offline_drafter/paths.sh"
TARGET_DIR=$(snapshot "$TARGET_MODEL" "$TARGET_REVISION")
export HF_HUB_OFFLINE=1
echo "job=$SLURM_JOB_ID node=$(hostname) DATASET=$DATASET target=$TARGET_DIR out=$ROLLOUTS gen_args=$GEN_ARGS commit=$(git -C "$REPO" rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
cd "$REPO"
python reproducibility/offline_drafter/gen_rollouts.py --dataset "$DATASET" --model "$TARGET_DIR" --out "$ROLLOUTS" \
    --dp "$(nvidia-smi -L | wc -l)" $GEN_ARGS "$@"
