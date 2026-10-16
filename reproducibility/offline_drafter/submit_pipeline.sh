#!/bin/bash
# Submit the offline-drafter pipeline for one or more datasets, stages chained with afterok dependencies:
#   [build_env] -> gen -> datagen -> train(+export)          (SQL additionally waits for [prepare_sql])
# Setup jobs are submitted only when their output is missing. Run from the repo root:
#   bash reproducibility/offline_drafter/submit_pipeline.sh dapo sql
#   TARGET=r1-distill-7b bash reproducibility/offline_drafter/submit_pipeline.sh dapo     (target presets: paths.sh)
# GPU_OPTS: sbatch options added to every GPU stage, e.g. H100 for long-context DAPO (v2 drafter):
#   GPU_OPTS="--gres=gpu:h100:4 --constraint=h100" TARGET=r1-distill-llama-8b NUM_PROMPTS=6000 GEN_MAX_NEW=32000 \
#   GEN_MAX_RUNNING=32 MAX_K=16 EAGLE_ROOT=<dir> bash reproducibility/offline_drafter/submit_pipeline.sh dapo
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
    case "$T:$DATASET" in r1-distill-7b:dapo|r1-distill-llama-8b:dapo) gen_opts="--gres=gpu:l40s:4 --mem=200G --time=08:00:00" ;; esac
    if [ "$DATASET" = sql ] && [ ! -f "$DATA_ROOT/SkyRL-SQL/train.parquet" ]; then
        j=$(sbatch --parsable $D/prepare_sql.sbatch); echo "prepare_sql: $j"; gen_dep="--dependency=afterok:$j"
    fi
    g=$(DATASET=$DATASET TARGET=$T sbatch --parsable -J eagle-gen-$T-$DATASET $gen_opts ${GPU_OPTS:-} ${GEN_TIME:+--time=$GEN_TIME} $gen_dep $D/gen.sbatch)
    d=$(DATASET=$DATASET TARGET=$T sbatch --parsable -J eagle-datagen-$T-$DATASET ${GPU_OPTS:-} ${DATAGEN_TIME:+--time=$DATAGEN_TIME} --dependency=afterok:$g$env_dep $D/datagen.sbatch)
    t=$(DATASET=$DATASET TARGET=$T sbatch --parsable -J eagle-train-$T-$DATASET ${GPU_OPTS:-} ${TRAIN_TIME:+--time=$TRAIN_TIME} --dependency=afterok:$d $D/train.sbatch)
    echo "$T $DATASET: gen $g -> datagen $d -> train $t"
done
squeue --me
