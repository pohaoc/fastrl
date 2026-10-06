#!/bin/bash
# Submit the offline-drafter pipeline for one or more datasets, stages chained with afterok dependencies:
#   [build_env] -> gen -> datagen -> train(+export)          (SQL additionally waits for [prepare_sql])
# Setup jobs are submitted only when their output is missing. Run from the repo root:
#   bash reproducibility/offline_drafter/submit_pipeline.sh dapo sql
#   TARGET=r1-distill-7b bash reproducibility/offline_drafter/submit_pipeline.sh dapo     (target presets: paths.sh)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
D=reproducibility/offline_drafter
source reproducibility/oscar/env.sh >/dev/null 2>&1 || true
EAGLE_ENV=${EAGLE_ENV:-$HOME/envs/eagle-train}

env_dep=""
if [ ! -x "$EAGLE_ENV/bin/python" ]; then
    j=$(sbatch --parsable $D/build_env.sbatch); echo "build_env: $j"; env_dep=":$j"
fi
T=${TARGET:-coder}
for DATASET in "$@"; do
    gen_dep=""
    gen_opts=""   # long reasoning traces: one engine replica per GPU on 4 GPUs
    [ "$T:$DATASET" = r1-distill-7b:dapo ] && gen_opts="--gres=gpu:l40s:4 --mem=200G --time=08:00:00"
    if [ "$DATASET" = sql ] && [ ! -f "$DATA_ROOT/SkyRL-SQL/train.parquet" ]; then
        j=$(sbatch --parsable $D/prepare_sql.sbatch); echo "prepare_sql: $j"; gen_dep="--dependency=afterok:$j"
    fi
    g=$(DATASET=$DATASET TARGET=$T sbatch --parsable -J eagle-gen-$T-$DATASET $gen_opts $gen_dep $D/gen.sbatch)
    d=$(DATASET=$DATASET TARGET=$T sbatch --parsable -J eagle-datagen-$T-$DATASET --dependency=afterok:$g$env_dep $D/datagen.sbatch)
    t=$(DATASET=$DATASET TARGET=$T sbatch --parsable -J eagle-train-$T-$DATASET --dependency=afterok:$d $D/train.sbatch)
    echo "$T $DATASET: gen $g -> datagen $d -> train $t"
done
squeue --me
