# Paths and per-dataset settings for the offline drafter pipeline; sourced by the .sbatch files here.
# DATASET=dapo|sql must be set.
: "${DATASET:?set DATASET=dapo or sql}"
source "$REPO/reproducibility/oscar/env.sh"          # HF_HOME, DATA_ROOT, SCRATCH_ROOT (and the fastrl env on PATH)
export EAGLE_ENV=${EAGLE_ENV:-$HOME/envs/eagle-train}
# Target (policy) model the drafter is trained for, pinned. The drafter's config is derived from the
# target's (export_drafter.py); its weights are trained on this model's hidden states.
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
TARGET_REVISION=${TARGET_REVISION:-c03e6d358207e414f1eca0bb1891e29f1db0e242}
TARGET_TAG=$(basename "$TARGET_MODEL")
# Warm start for the trainable drafter weights (fc + decoder layer); INIT_DRAFT=none trains from scratch.
# embed_tokens / lm_head always come frozen from the target, whatever the warm start.
INIT_DRAFT=${INIT_DRAFT:-mit-han-lab/Qwen2.5-7B-Eagle-RL}
INIT_REVISION=${INIT_REVISION:-38905764986fdd43c64f0b5ab2a0bdac6bf71b08}
# The released EAGLE checkpoint whose tensor names the export reproduces.
LAYOUT_REF=mit-han-lab/Qwen2.5-7B-Eagle-RL
LAYOUT_REVISION=38905764986fdd43c64f0b5ab2a0bdac6bf71b08
# One tree per target, so drafters for different targets never overwrite each other's data.
EAGLE_ROOT=${EAGLE_ROOT:-$SCRATCH_ROOT/eagle/$TARGET_TAG}
ROLLOUTS=$EAGLE_ROOT/rollouts/$DATASET               # gen_rollouts.py output (data/train.parquet, stats.json)
case $DATASET in
  dapo) MAX_K=4;  MICRO=2; ACCUM=2; EPOCHS=${EPOCHS:-5}; TRAIN_MAX_LEN=${TRAIN_MAX_LEN:-}  ;;   # ~9k sequences <=4K tokens
  # ~2.6k trajectories <=12K tokens (schema prompts). Training truncates to 8K: at 12K the loss's
  # [len x 152064] fp32 tensors (~7 GB each) OOM a 44 GB L40S; 8K kept 94% of generated tokens for Qwen2.5-7B.
  sql)  MAX_K=12; MICRO=1; ACCUM=4; EPOCHS=${EPOCHS:-10}; TRAIN_MAX_LEN=${TRAIN_MAX_LEN:-8192} ;;
  *) echo "DATASET must be dapo or sql"; exit 1 ;;
esac
# eagle_trainer.py reads the padding length from the data directory's name (<name>-<N>K)
HS_ROOT=$EAGLE_ROOT/hidden_states/$DATASET
HS_DIR=$HS_ROOT/$DATASET-${MAX_K}K
CKPT_DIR=$EAGLE_ROOT/ckpt/$DATASET                   # DeepSpeed checkpoints, one per epoch
EXPORT_DIR=$EAGLE_ROOT/drafters/$TARGET_TAG-Eagle-$DATASET   # HF format: use as SPEC_MODEL_PATH

# Local snapshot dir of a pinned revision (downloads on first use).
snapshot() { python -c "from huggingface_hub import snapshot_download as s; print(s('$1', revision='$2'))"; }
