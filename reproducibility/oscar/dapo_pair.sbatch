#!/bin/bash
# Paired DAPO-Math RL comparison on ONE 4x H100 node, arms run back to back (same node = no node-speed confound):
#   1. sdon  : FastRL adaptive speculative decoding with our offline-trained R1-Distill-Llama-8B drafter
#   2. sdoff : speculative.enable=false (TLT baseline, speculation off)
# Identical settings otherwise (run_grpo_dapo.sh, which matches the SysX DAPO sync configuration on every shared
# parameter); same seed, so step 1 sees the same prompts.
#   STEPS=10 sbatch reproducibility/oscar/dapo_pair.sbatch       (submit from the repo root)
# Per-arm logs: $RUN_ROOT/dapo_pair/<jobid>/<arm>.log; compare with reproducibility/oscar/compare_runs.py.
#SBATCH --job-name=dapo-pair-h100
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=20:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -uo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source reproducibility/oscar/env.sh
ulimit -u "$(ulimit -Hu)"   # Ray + SGLang + torch threads; same as the SQL pair
export STEPS=${STEPS:-10}
DRAFTER=${DRAFTER:-$SCRATCH_ROOT/eagle/DeepSeek-R1-Distill-Llama-8B/drafters/DeepSeek-R1-Distill-Llama-8B-Eagle-dapo}
BASE=$RUN_ROOT/dapo_pair/$SLURM_JOB_ID
mkdir -p "$BASE"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) commit=$(git rev-parse --short HEAD) STEPS=$STEPS drafter=$DRAFTER"
git status --short
python -c "import verl, sglang; print('verl from', verl.__file__); print('sglang from', sglang.__file__)"
gpu_preflight
cat "$DRAFTER/export_info.json" | head -5

FAILED=0
arm() {  # name SD [Hydra overrides...]
    local name=$1 sd=$2
    shift 2
    export RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID-$name
    # Per-iteration scheduler log inside the RL rollouts (sglang/srt/fastrl_steplog.py; Ray workers inherit it):
    # acceptance and time by running batch size for every RL step. Same cost in both arms.
    export FASTRL_STEPLOG_DIR=$BASE/$name/steplog
    mkdir -p "$RAY_TMPDIR" "$FASTRL_STEPLOG_DIR"
    echo; echo "=================== $name (SD=$sd) $(date +%T)"
    SD=$sd SPEC_MODEL_PATH=$DRAFTER OUT=$BASE/$name NAME=r1llama8b-dapo-$name \
        bash reproducibility/oscar/run_grpo_dapo.sh "$@" > "$BASE/$name.log" 2>&1
    local rc=$?
    echo "=================== $name exit=$rc $(date +%T)"
    if [ $rc -ne 0 ]; then
        FAILED=1
        tail -40 "$BASE/$name.log"
        cp -r "$RAY_TMPDIR"/ray/session_latest/logs "$BASE/$name.ray_logs" 2>/dev/null || true
    fi
    sleep 10   # let the arm's Ray cluster finish tearing down before the next starts
}
arm sdon true "$@"
arm sdoff false "$@"
python reproducibility/oscar/compare_runs.py "$BASE/sdon.log" "$BASE/sdoff.log"
echo "end=$(date)"
exit $FAILED
