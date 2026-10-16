# Paths and per-dataset settings for the offline drafter pipeline; sourced by the .sbatch files here.
# DATASET=dapo|sql must be set.
: "${DATASET:?set DATASET=dapo or sql}"
source "$REPO/reproducibility/oscar/env.sh"          # HF_HOME, DATA_ROOT, SCRATCH_ROOT (and the fastrl env on PATH)
export EAGLE_ENV=${EAGLE_ENV:-$HOME/envs/eagle-train}
# Target (policy) model the drafter is trained for, pinned. The drafter's config is derived from the
# target's (export_drafter.py); its weights are trained on this model's hidden states.
#   TARGET=coder          Qwen/Qwen2.5-Coder-7B-Instruct (SkyRL-SQL)
#   TARGET=r1-distill-7b  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (DAPO-Math-17k)
#   TARGET=r1-distill-llama-8b  deepseek-ai/DeepSeek-R1-Distill-Llama-8B (DAPO-Math-17k, as the SysX DAPO runs)
# TARGET_MODEL / TARGET_REVISION override the preset.
TARGET=${TARGET:-coder}
# _I: default warm start (none = random init); _L: released drafter whose tensor names the export reproduces
# (none = the checkpoint's own names; there is no released FastRL drafter for Llama targets).
case $TARGET in
  coder)         _M=Qwen/Qwen2.5-Coder-7B-Instruct;           _R=c03e6d358207e414f1eca0bb1891e29f1db0e242
                 _I=mit-han-lab/Qwen2.5-7B-Eagle-RL; _L=mit-han-lab/Qwen2.5-7B-Eagle-RL ;;
  r1-distill-7b) _M=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B;   _R=916b56a44061fd5cd7d6a8fb632557ed4f724f60
                 _I=mit-han-lab/Qwen2.5-7B-Eagle-RL; _L=mit-han-lab/Qwen2.5-7B-Eagle-RL ;;
  r1-distill-llama-8b) _M=deepseek-ai/DeepSeek-R1-Distill-Llama-8B; _R=6a6f4aa4197940add57724a7707d069478df56b1
                 _I=none; _L=none ;;
  *) echo "TARGET must be coder, r1-distill-7b or r1-distill-llama-8b"; exit 1 ;;
esac
TARGET_MODEL=${TARGET_MODEL:-$_M}
TARGET_REVISION=${TARGET_REVISION:-$_R}
TARGET_TAG=$(basename "$TARGET_MODEL")
# Warm start for the trainable drafter weights (fc + decoder layer); INIT_DRAFT=none trains from scratch.
# embed_tokens / lm_head always come frozen from the target, whatever the warm start.
INIT_DRAFT=${INIT_DRAFT:-$_I}
INIT_REVISION=${INIT_REVISION:-38905764986fdd43c64f0b5ab2a0bdac6bf71b08}
LAYOUT_REF=$_L
LAYOUT_REVISION=38905764986fdd43c64f0b5ab2a0bdac6bf71b08
# One tree per target, so drafters for different targets never overwrite each other's data.
EAGLE_ROOT=${EAGLE_ROOT:-$SCRATCH_ROOT/eagle/$TARGET_TAG}
ROLLOUTS=$EAGLE_ROOT/rollouts/$DATASET               # gen_rollouts.py output (data/train.parquet, stats.json)
# GEN_ARGS: gen_rollouts.py settings; they must match the RL rollouts this drafter will serve.
case $DATASET:$TARGET in
  dapo:r1-distill-7b)
    # R1-Distill writes long <think> traces: sample its full response (DeepSeek's recommended T 0.6 / top-p 0.95)
    # but store and train on the first 8K tokens of each row (datagen truncates; ~7.2 KB/token on disk, so
    # 2,000 rows <= 118 GB). 4 GPUs, one worker process each (gen.sbatch).
    NUM_PROMPTS=${NUM_PROMPTS:-2000}; GEN_MAX_NEW=${GEN_MAX_NEW:-16384}
    # --max_running 24: 24 x (16K + prompt) fits the ~429K-token KV pool of one L40S at mem 0.85, so no retraction.
    GEN_ARGS="--num_prompts $NUM_PROMPTS --max_new_tokens $GEN_MAX_NEW --temperature ${GEN_TEMP:-0.6} --top_p ${GEN_TOP_P:-0.95} --max_running 24"
    MAX_K=8; MICRO=1; ACCUM=4; EPOCHS=${EPOCHS:-10}; TRAIN_MAX_LEN=${TRAIN_MAX_LEN:-} ;;
  dapo:r1-distill-llama-8b)
    # Matches the SysX DAPO runs (~/scratch/runs/dapo-b128-sync-8065792/run_config.toml): their dapo_math17k
    # (14,116 prompts), prompts <= 1024 tokens, T 1.0 / top-p 1.0. Their responses run to 32000 tokens; training uses
    # only the first 8K tokens of each row, and the first 8K tokens of a response have the same distribution
    # whatever the cap, so generation stops at 8K (about 4x faster). --max_running 16: Llama-8B KV is 128 KiB per
    # token, ~165K tokens per L40S at mem 0.85, and 16 x (8192 + 1024) fits without retraction.
    NUM_PROMPTS=${NUM_PROMPTS:-2000}; GEN_MAX_NEW=${GEN_MAX_NEW:-8192}
    GEN_ARGS="--dapo_parquet /oscar/data/deeptir/mborjigi/datasets/dapo_math17k/train.parquet --max_prompt_tokens 1024"
    # v2 (2026-10-16, on H100): NUM_PROMPTS=6000 GEN_MAX_NEW=32000 GEN_MAX_RUNNING=32 MAX_K=16. The 8K v1 drafter
    # (top-1 0.68, 36% of its rollouts cut at 8K) accepted ~1.96 tokens per verify step in DAPO, slower than
    # plain decoding. 16K windows need H100: 16K x 128,256-vocab fp32 loss tensors OOM a 44 GB L40S.
    GEN_ARGS="$GEN_ARGS --num_prompts $NUM_PROMPTS --max_new_tokens $GEN_MAX_NEW --temperature ${GEN_TEMP:-1.0} --top_p ${GEN_TOP_P:-1.0} --max_running ${GEN_MAX_RUNNING:-16}"
    MAX_K=${MAX_K:-8}; MICRO=1; ACCUM=4; EPOCHS=${EPOCHS:-10}; TRAIN_MAX_LEN=${TRAIN_MAX_LEN:-} ;;
  dapo:*)
    GEN_ARGS="${GEN_ARGS:-}"
    MAX_K=4;  MICRO=2; ACCUM=2; EPOCHS=${EPOCHS:-5}; TRAIN_MAX_LEN=${TRAIN_MAX_LEN:-}  ;;   # ~9k sequences <=4K tokens
  # ~2.6k trajectories <=12K tokens (schema prompts). Training truncates to 8K: at 12K the loss's
  # [len x 152064] fp32 tensors (~7 GB each) OOM a 44 GB L40S; 8K kept 94% of generated tokens for Qwen2.5-7B.
  sql:*)
    GEN_ARGS="${GEN_ARGS:-}"
    MAX_K=12; MICRO=1; ACCUM=4; EPOCHS=${EPOCHS:-10}; TRAIN_MAX_LEN=${TRAIN_MAX_LEN:-8192} ;;
  *) echo "DATASET must be dapo or sql"; exit 1 ;;
esac
# eagle_trainer.py reads the padding length from the data directory's name (<name>-<N>K)
HS_ROOT=$EAGLE_ROOT/hidden_states/$DATASET
HS_DIR=$HS_ROOT/$DATASET-${MAX_K}K
CKPT_DIR=$EAGLE_ROOT/ckpt/$DATASET                   # DeepSpeed checkpoints, newest 2 epochs kept (--keep_ckpts)
EXPORT_DIR=$EAGLE_ROOT/drafters/$TARGET_TAG-Eagle-$DATASET   # HF format: use as SPEC_MODEL_PATH

# Local snapshot dir of a pinned revision (downloads on first use).
snapshot() { python -c "from huggingface_hub import snapshot_download as s; print(s('$1', revision='$2'))"; }
