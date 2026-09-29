# Paths and per-dataset settings for the offline drafter pipeline; sourced by the .sbatch files here.
# DATASET=dapo|sql must be set.
: "${DATASET:?set DATASET=dapo or sql}"
source "$REPO/reproducibility/oscar/env.sh"          # HF_HOME, DATA_ROOT, SCRATCH_ROOT (and the fastrl env on PATH)
export EAGLE_ENV=${EAGLE_ENV:-$HOME/envs/eagle-train}
EAGLE_ROOT=${EAGLE_ROOT:-$SCRATCH_ROOT/eagle}
ROLLOUTS=$EAGLE_ROOT/rollouts/$DATASET               # gen_rollouts.py output (data/train.parquet, stats.json)
case $DATASET in
  dapo) MAX_K=4;  MICRO=2; ACCUM=2; EPOCHS=${EPOCHS:-5}  ;;   # ~9k sequences <=4K tokens
  sql)  MAX_K=12; MICRO=1; ACCUM=4; EPOCHS=${EPOCHS:-10} ;;   # ~2.6k trajectories <=12K tokens (schema prompts)
  *) echo "DATASET must be dapo or sql"; exit 1 ;;
esac
# eagle_trainer.py reads the padding length from the data directory's name (<name>-<N>K)
HS_ROOT=$EAGLE_ROOT/hidden_states/$DATASET
HS_DIR=$HS_ROOT/$DATASET-${MAX_K}K
CKPT_DIR=$EAGLE_ROOT/ckpt/$DATASET                   # DeepSpeed checkpoints, one per epoch
EXPORT_DIR=$EAGLE_ROOT/drafters/Qwen2.5-7B-Eagle-RL-$DATASET   # HF format: use as spec_model_path / EAGLE_PATH
INIT_DRAFT=${INIT_DRAFT:-mit-han-lab/Qwen2.5-7B-Eagle-RL}
TARGET_MODEL=Qwen/Qwen2.5-7B
