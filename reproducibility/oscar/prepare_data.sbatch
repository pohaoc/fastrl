#!/bin/bash
# Build $DATA_ROOT (see env.sh) for run_grpo_7B_4gpu.sh on Oscar:
#   DAPO-Math-17k/     deduplicated train + AIME-2024 validation (prepare_dapo.py)
#   Eurus-2-RL-Data -> ../eurus/raw   (full PRIME-RL/Eurus-2-RL-Data; the top-level eurus/*.parquet is a
#                                      math-only subset and must not be used)
# SkyRL-SQL is not built here (prepare_sql.py downloads a 22 GB bundle); not needed for the drafter runs.
# Submit from the repo root: sbatch reproducibility/oscar/prepare_data.sbatch
#SBATCH --job-name=fastrl-prep-data
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
[ -f "$REPO/reproducibility/dataset/prepare_dapo.py" ] || { echo "submit from the repo root (or set FASTRL_REPO)"; exit 1; }
source "$REPO/reproducibility/oscar/env.sh"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

python "$REPO/reproducibility/dataset/prepare_dapo.py" \
    --out "$DATA_ROOT/DAPO-Math-17k" --raw "$DATA_ROOT/DAPO-Math-17k-raw"

EURUS_RAW=$SCRATCH_ROOT/data/eurus/raw
[ -e Eurus-2-RL-Data ] || ln -s "$EURUS_RAW" Eurus-2-RL-Data

python - <<'EOF'
import pandas as pd
# Expected: DAPO 17,398 / 30 (HANDOFF: deduped); Eurus 480,537 / 2,048 (prepare_eurus.sh).
for d in ("DAPO-Math-17k", "Eurus-2-RL-Data"):
    for s in ("train", "validation"):
        print(f"{d}/{s}: {len(pd.read_parquet(f'{d}/{s}.parquet')):,} rows")
EOF
ls -l "$DATA_ROOT"
