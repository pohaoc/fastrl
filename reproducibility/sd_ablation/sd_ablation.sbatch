#!/bin/bash
# Rollout-only ablation of the SD-on slowdown on SkyRL-SQL (pair job 6995508: SD-on rollout 1.14x slower).
# Replays one RL rollout step (256 prompts x 5, step-1 settings) per arm, all arms back to back on ONE node:
#   off            verl SD off (sglang defaults)                                   = RL sdoff arm
#   off_tax        SD off + the 3 engine settings verl adds when SD is on          (engine-settings tax)
#   off_mrr512     SD off + max_running_requests=512                               (one factor at a time)
#   off_cg32       SD off + cuda_graph_max_bs=32
#   off_nooverlap  SD off + disable_overlap_schedule
#   sd_gated       SD engine + tuned drafter, never speculates on decode           (cost of the SD machinery)
#   sd_tuned       verl SD on, tuned drafter                                       = RL sdon arm
#   sd_released    verl SD on, mit-han-lab/Qwen2.5-7B-Eagle-RL                     (drafter quality)
#   sd_tuned_rep, off_rep  replicates (sampling noise + drift over the job)
#   sbatch reproducibility/sd_ablation/sd_ablation.sbatch      (submit from the repo root)
# Summary at the end of the log (report.py); per-arm result.json + step logs under $RUN_ROOT/sd_ablation/<jobid>/.
#SBATCH --job-name=sd-ablation-l40s
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu3001                 # uncorrectable ECC errors (2026-09-30, 2026-10-04)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -uo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source reproducibility/oscar/env.sh
OUT=$RUN_ROOT/sd_ablation/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) commit=$(git rev-parse --short HEAD)"
git status --short
python -c "import verl, sglang; print('verl from', verl.__file__); print('sglang from', sglang.__file__)"
gpu_preflight

snap() { python -c "from huggingface_hub import snapshot_download as s; print(s('$1', revision='$2'))"; }
MODEL=$(snap Qwen/Qwen2.5-Coder-7B-Instruct c03e6d358207e414f1eca0bb1891e29f1db0e242)
RELEASED=$(snap mit-han-lab/Qwen2.5-7B-Eagle-RL 38905764986fdd43c64f0b5ab2a0bdac6bf71b08)
TUNED=${TUNED:-$SCRATCH_ROOT/eagle/Qwen2.5-Coder-7B-Instruct/drafters/Qwen2.5-Coder-7B-Instruct-Eagle-sql}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true

arm() {  # name, then sd_ablation.py args
    local name=$1; shift
    echo; echo "=================== $name: $* ($(date +%T))"
    # Own process group, so a dead or hung engine's scheduler processes are reaped with it (30 min cap).
    setsid python reproducibility/sd_ablation/sd_ablation.py --out "$OUT/$name" --model "$MODEL" "$@" \
        > "$OUT/$name.log" 2>&1 &
    local pid=$!
    ( sleep 1800; echo "TIMEOUT $name"; kill -TERM -- -$pid ) 2>/dev/null & local dog=$!
    wait $pid; local rc=$?
    kill $dog 2>/dev/null; kill -9 -- -$pid 2>/dev/null; sleep 5
    grep -a "^RESULT" "$OUT/$name.log" || tail -30 "$OUT/$name.log"
    echo "=================== $name exit=$rc ($(date +%T))"
}

arm off
arm off_tax       --mrr 512 --cg_max_bs 32 --no_overlap 1
arm sd_tuned      --sd 1 --drafter "$TUNED"
arm sd_gated      --sd 1 --drafter "$TUNED" --force_plain 1
arm off_mrr512    --mrr 512
arm off_cg32      --cg_max_bs 32
arm off_nooverlap --no_overlap 1
arm sd_released   --sd 1 --drafter "$RELEASED"
arm sd_tuned_rep  --sd 1 --drafter "$TUNED"
arm off_rep

echo; echo "=================== SUMMARY"
python reproducibility/sd_ablation/report.py "$OUT" --ref off
echo "end=$(date)"
