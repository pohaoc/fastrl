#!/bin/bash
# Paired SkyRL-SQL RL comparison on ONE 4x L40S node, arms run back to back (same node = no node-speed confound):
#   1. sdon  : FastRL adaptive speculative decoding with our offline-trained Coder-7B-Instruct drafter
#   2. sdoff : speculative.enable=false (TLT baseline, speculation off)
# Identical settings otherwise (run_grpo_sql.sh, which matches the SysX SQL sync configuration on every shared
# parameter); same seed, so step 1 sees the same prompts.
#   STEPS=10 sbatch reproducibility/oscar/sql_pair.sbatch       (submit from the repo root)
#   35 steps: STEPS=35 sbatch --time=32:00:00 reproducibility/oscar/sql_pair.sbatch  (~14 h per arm on L40S)
# Per-arm logs: $RUN_ROOT/sql_pair/<jobid>/<arm>.log; compare with reproducibility/oscar/compare_runs.py.
#SBATCH --job-name=sql-pair-l40s
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu3001                 # uncorrectable ECC errors (2026-09-30, 2026-10-04)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=14:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -uo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source reproducibility/oscar/env.sh
# 2048 SkyRL env workers (the SysX SQL setting) exceed the default soft thread limit (4096 incl. Ray/SGLang/torch).
ulimit -u "$(ulimit -Hu)"
export STEPS=${STEPS:-10}
DRAFTER=${DRAFTER:-$SCRATCH_ROOT/eagle/Qwen2.5-Coder-7B-Instruct/drafters/Qwen2.5-Coder-7B-Instruct-Eagle-sql}
BASE=$RUN_ROOT/sql_pair/$SLURM_JOB_ID
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
    SD=$sd SPEC_MODEL_PATH=$DRAFTER OUT=$BASE/$name NAME=coder7b-sql-$name \
        bash reproducibility/oscar/run_grpo_sql.sh "$@" > "$BASE/$name.log" 2>&1
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
