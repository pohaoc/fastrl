#!/bin/bash
# Standalone check: does FastRL's EAGLE engine run, and run correctly, with attention_backend=fa3 on 4x L40S?
# Each variant is a fresh process built with verl's in-loop engine arguments (see fa3_check.py).
#   sbatch reproducibility/oscar/fa3_check.sbatch          (submit from the repo root)
# Summary at the end of the log; per-variant result.json under $RUN_ROOT/fa3_check/<jobid>/.
#SBATCH --job-name=fa3-check-l40s
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu3001                 # uncorrectable ECC errors (2026-09-30, 2026-10-04)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -uo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
source "$REPO/reproducibility/oscar/env.sh"
OUT=$RUN_ROOT/fa3_check/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) commit=$(git -C "$REPO" rev-parse --short HEAD)"
git -C "$REPO" status --short
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

# Pinned HF revisions (main as of 2026-10-03; Qwen2.5-7B / Eagle-RL as in earlier runs).
CODER=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-Coder-7B-Instruct', revision='c03e6d358207e414f1eca0bb1891e29f1db0e242'))")
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-7B', revision='d149729398750b98c0af14eb82c78cfe92750796'))")
EAGLE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('mit-han-lab/Qwen2.5-7B-Eagle-RL', revision='38905764986fdd43c64f0b5ab2a0bdac6bf71b08'))")
echo "CODER=$CODER"; echo "BASE=$BASE"; echo "EAGLE=$EAGLE"
export HF_HUB_OFFLINE=1

run() {  # name, then fa3_check.py args
    local name=$1; shift
    echo; echo "=================== $name: $* ($(date +%T))"
    # Own process group, so a dead or hung engine's scheduler children are reaped with it (15 min cap).
    setsid python "$REPO/reproducibility/oscar/fa3_check.py" --out "$OUT/$name" --drafter "$EAGLE" "$@" &
    local pid=$!
    ( sleep 900; echo "TIMEOUT $name"; kill -TERM -- -$pid ) 2>/dev/null & local dog=$!
    wait $pid; local rc=$?
    kill $dog 2>/dev/null; kill -9 -- -$pid 2>/dev/null; sleep 5
    echo "=================== $name exit=$rc ($(date +%T))"
}

# Target config for the SQL runs: Coder-7B-Instruct (max_position 32768), upstream mem 0.4.
run coder-fa3-plain-tp2  --model "$CODER" --backend fa3    --sd 0 --tp 2
run coder-fa3-sd-tp2     --model "$CODER" --backend fa3    --sd 1 --tp 2
run coder-triton-sd-tp2  --model "$CODER" --backend triton --sd 1 --tp 2
run coder-fa3-sd-tp4     --model "$CODER" --backend fa3    --sd 1 --tp 4
# The configuration that crashed in-loop before (jobs 6747961/6749022): Qwen2.5-7B base, whose
# max_position 131072 sizes fa3's draft graph buffers, TP=4, mem 0.6. Does it crash standalone?
run base-fa3-sd-tp4-m06  --model "$BASE"  --backend fa3    --sd 1 --tp 4 --mem 0.6

echo; echo "=================== SUMMARY (greedy agreement is vs coder-fa3-plain-tp2; the base run has a different target)"
python "$REPO/reproducibility/oscar/fa3_check.py" --summarize "$OUT" --ref coder-fa3-plain-tp2
echo "end=$(date)"
