#!/bin/bash
# One SkyRL-SQL RL run (run_grpo_sql.sh) on 4x L40S. For the SD-on vs SD-off comparison use sql_pair.sbatch.
#   SD=true SPEC_MODEL_PATH=<drafter dir> STEPS=1 sbatch reproducibility/oscar/grpo_sql.sbatch   (from the repo root)
# Extra arguments are Hydra overrides.
#SBATCH --job-name=grpo-sql-l40s
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source reproducibility/oscar/env.sh
export SD=${SD:-true}
export OUT=$RUN_ROOT/grpo_sql/$SLURM_JOB_ID NAME=coder7b-sql-sd${SD}
export RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID
mkdir -p "$RAY_TMPDIR"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse --short HEAD)"; git status --short
python -c "import verl, sglang; print('verl from', verl.__file__); print('sglang from', sglang.__file__)"
set +e
bash reproducibility/oscar/run_grpo_sql.sh "$@"
rc=$?
[ $rc -ne 0 ] && cp -r "$RAY_TMPDIR"/ray/session_latest/logs "$OUT/ray_logs" 2>/dev/null
echo "exit=$rc end=$(date)"; exit $rc
