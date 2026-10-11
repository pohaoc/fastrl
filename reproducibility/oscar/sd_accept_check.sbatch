#!/bin/bash
# Why does the SQL SD arm accept ~0 draft tokens inside RL (job 7283745) when the same engine settings accept 3.9
# standalone (sweep 7226017)? Four single-GPU engines (TP 1, mem 0.70, SD on), one per GPU, same greedy prompts:
#   plain       : the sweep's path (target restored from its CPU backup, no push)            -> expect ~3.9
#   rl          : verl's path: release all after start-up, resume weights, push target, resume KV
#   rl_nopush   : verl's order, but the target comes back from its CPU backup (isolates the push)
#   rl_noreldraft: verl's path with FASTRL_RELEASE_SD_MEMORY=0 (drafter stays resident; isolates our release)
#   sbatch reproducibility/oscar/sd_accept_check.sbatch       (submit from the repo root)
#SBATCH --job-name=sd-accept-check
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu3001                 # uncorrectable ECC errors (2026-09-30, 2026-10-04)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=00:45:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -uo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source reproducibility/oscar/env.sh
OUT=$RUN_ROOT/sd_accept_check/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) commit=$(git rev-parse --short HEAD)"
git status --short
gpu_preflight
MODEL=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-Coder-7B-Instruct', revision='c03e6d358207e414f1eca0bb1891e29f1db0e242'))")
DRAFTER=$SCRATCH_ROOT/eagle/Qwen2.5-Coder-7B-Instruct/drafters/Qwen2.5-Coder-7B-Instruct-Eagle-sql
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true

run_on() {  # gpu name release_sd_memory [sd_memory_check args...]  (own process group, 15 min cap)
    local gpu=$1 name=$2 rel=$3
    shift 3
    FASTRL_RELEASE_SD_MEMORY=$rel CUDA_VISIBLE_DEVICES=$gpu setsid python reproducibility/oscar/sd_memory_check.py \
        --out "$OUT/$name.json" --model "$MODEL" --drafter "$DRAFTER" --sd 1 --tp 1 --mem 0.70 \
        --chunked_prefill 8192 --smi_index "$gpu" --n_batch 32 --cycles 1 "$@" > "$OUT/$name.log" 2>&1 &
    local pid=$!
    ( sleep 900; echo "TIMEOUT $name"; kill -TERM -- -$pid; sleep 20; kill -9 -- -$pid ) 2>/dev/null & local dog=$!
    wait $pid; local rc=$?
    kill $dog 2>/dev/null; kill -9 -- -$pid 2>/dev/null
    echo "$name exit=$rc"
}
run_on 0 plain 1 &
run_on 1 rl 1 --rl_wakeup 1 &
run_on 2 rl_nopush 1 --rl_wakeup 1 --push_target 0 &
run_on 3 rl_noreldraft 0 --rl_wakeup 1 &
wait

echo; echo "=================== SUMMARY (accept = generated tokens per verify step, greedy, 512 tokens x 6 prompts)"
for n in plain rl rl_nopush rl_noreldraft; do
    python -c "import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], r.get('phase'), 'before', r.get('accept_before'), 'after', r.get('accept_after'), 'greedy same', r.get('greedy_identical'), '/', r.get('greedy_total'))" "$OUT/$n.json" "$n" 2>/dev/null || echo "$n no result"
done
echo "end=$(date)"
