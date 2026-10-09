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
G="python reproducibility/offline_drafter/gen_rollouts.py --dataset $DATASET --model $TARGET_DIR --out $ROLLOUTS"
if [ "$DATASET" = dapo ]; then
    # One independent worker per GPU; each writes 100-prompt chunk files as they finish. A crashed worker loses
    # only its unfinished chunk: pass 2 regenerates whatever is missing, then the chunks are merged.
    NGPU=$(nvidia-smi -L | wc -l)
    mkdir -p "$ROLLOUTS"   # the worker logs below are redirected into it before any worker creates it (job 7183332)
    merged=0
    for pass in 1 2; do
        for i in $(seq 0 $((NGPU - 1))); do
            CUDA_VISIBLE_DEVICES=$i $G --shard "$i" --num_shards "$NGPU" $GEN_ARGS "$@" \
                > "$ROLLOUTS/worker$i.pass$pass.log" 2>&1 &
        done
        wait
        grep -h "chunk .* done\|chunks to generate" "$ROLLOUTS"/worker*.pass$pass.log | tail -n 8
        if $G --merge $GEN_ARGS "$@"; then merged=1; break; fi
        echo "pass $pass left chunks missing; worker log tails:"; tail -n 5 "$ROLLOUTS"/worker*.pass$pass.log
    done
    [ "$merged" = 1 ] || { echo "chunks still missing after 2 passes"; exit 1; }
else
    $G $GEN_ARGS "$@"
fi
