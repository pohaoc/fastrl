#!/bin/bash
# Build $DATA_ROOT/SkyRL-SQL (verl format + the SQLite databases it references) from the SkyRL-SQL-653 parquets
# and the OmniSQL data.zip already on scratch; nothing is downloaded. Needed by gen_rollouts.py --dataset sql
# and by SQL RL runs (DATASET=sql in run_grpo_7B_4gpu.sh).
# Submit from the repo root: sbatch reproducibility/offline_drafter/prepare_sql.sbatch
#SBATCH --job-name=prep-sql
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
source "$REPO/reproducibility/oscar/env.sh"
SRC=${SQL_SRC:-$SCRATCH_ROOT/data/text2sql}   # train.parquet, validation.parquet (SkyRL newfmt), data.zip (OmniSQL)
cd "$REPO"
python reproducibility/dataset/prepare_sql.py --out "$DATA_ROOT/SkyRL-SQL" --raw "$SRC" --zip "$SRC/data.zip"
du -sh "$DATA_ROOT/SkyRL-SQL/db"
