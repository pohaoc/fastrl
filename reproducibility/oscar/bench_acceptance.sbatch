#!/bin/bash
# Standalone drafter acceptance on Oscar (reproducibility/dataset/bench_acceptance.sh), sized for
# gpu-debug (QOS limit per user: 1 h, 4 GPUs, 8 CPUs, 96G). It checks the env and this checkout's SGLang
# on H100 with fa3 EAGLE, and gives acceptance at the chosen TP outside the RL loop.
# Submit from the repo root (DAPO data from prepare_data.sbatch must exist):
#   sbatch reproducibility/oscar/bench_acceptance.sbatch                       # DAPO, TP=1
#   DATASET=eurus TP=2 sbatch --gres=gpu:h100:2 reproducibility/oscar/bench_acceptance.sbatch
#SBATCH --job-name=fastrl-bench-accept
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
[ -f "$REPO/reproducibility/dataset/bench_acceptance.sh" ] || { echo "submit from the repo root (or set FASTRL_REPO)"; exit 1; }
source "$REPO/reproducibility/oscar/env.sh"
export HF_HUB_OFFLINE=1
export DATASET=${DATASET:-dapo} TP=${TP:-1}
export OUT=${OUT:-$RUN_ROOT/bench_acceptance/$SLURM_JOB_ID}
mkdir -p "$OUT"

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) DATASET=$DATASET TP=$TP OUT=$OUT" \
     "EAGLE_PATH=${EAGLE_PATH:-default} PROMPTS_PARQUET=${PROMPTS_PARQUET:-default}"
echo "commit=$(git -C "$REPO" rev-parse HEAD)"; git -C "$REPO" status --short
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import torch, sglang, verl; print('torch', torch.__version__, 'sglang', sglang.__file__, 'verl', verl.__file__)"

# EAGLE_PATHS="<d1> <d2> ...": benchmark several drafters on the same prompts in one job (default: EAGLE_PATH)
for EAGLE_PATH in ${EAGLE_PATHS:-${EAGLE_PATH:-mit-han-lab/Qwen2.5-7B-Eagle-RL}}; do
    export EAGLE_PATH
    echo "===== drafter $EAGLE_PATH $(date)"
    bash "$REPO/reproducibility/dataset/bench_acceptance.sh"
done
echo "end=$(date); full log in $OUT/bench/"
