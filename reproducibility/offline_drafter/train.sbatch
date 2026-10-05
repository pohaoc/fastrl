#!/bin/bash
# Stage 3: train the EAGLE drafter for $TARGET_MODEL on its cached hidden states (warm start: $INIT_DRAFT, or
# INIT_DRAFT=none for scratch; embed_tokens / lm_head are the target's, frozen)
# (eagle-train/eagle_trainer.py, DeepSpeed ZeRO-2, 4x L40S), then export the last epoch to HF format.
# Resumes from $CKPT_DIR/latest if a previous run was cut off.
# Submit from the repo root: DATASET=dapo|sql sbatch reproducibility/offline_drafter/train.sbatch
# Knobs (env): EPOCHS (dapo 5, sql 10), LR (2e-5 peak, linear decay to 0), WARMUP (50 optimizer steps).
#SBATCH --job-name=eagle-train
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu3001                 # uncorrectable ECC errors (2026-09-30, 2026-10-04)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
source "$REPO/reproducibility/offline_drafter/paths.sh"
source "$EAGLE_ENV/bin/activate"
export WANDB_MODE=offline WANDB_DIR=$CKPT_DIR TOKENIZERS_PARALLELISM=true NCCL_DEBUG=WARN
LR=${LR:-2e-5}
WARMUP=${WARMUP:-50}
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p "$CKPT_DIR"
TARGET_DIR=$(snapshot "$TARGET_MODEL" "$TARGET_REVISION")
INIT_DIR=; [ "$INIT_DRAFT" = none ] || INIT_DIR=$(snapshot "$INIT_DRAFT" "$INIT_REVISION")
LAYOUT_DIR=$(snapshot "$LAYOUT_REF" "$LAYOUT_REVISION")
export HF_HUB_OFFLINE=1
N_FILES=$(ls "$HS_DIR" | grep -c '^data_.*\.pt$')
N_TRAIN=$((N_FILES * 95 / 100))                        # eagle_trainer keeps 5% for validation
STEPS_PER_EPOCH=$(( (N_TRAIN + MICRO * NGPU * ACCUM - 1) / (MICRO * NGPU * ACCUM) ))
TOTAL=$((STEPS_PER_EPOCH * EPOCHS))
CFG=$CKPT_DIR/ds_config.json
cat > "$CFG" <<EOF
{
  "bf16": {"enabled": true},
  "optimizer": {"type": "AdamW",
                "params": {"lr": $LR, "weight_decay": 0.0, "betas": [0.9, 0.95], "adam_w_mode": true, "torch_adam": true}},
  "scheduler": {"type": "WarmupDecayLR",
                "params": {"warmup_min_lr": 0, "warmup_max_lr": $LR, "warmup_num_steps": $WARMUP, "total_num_steps": $TOTAL}},
  "zero_optimization": {"stage": 2, "allgather_partitions": true, "allgather_bucket_size": 2e8, "overlap_comm": true,
                        "reduce_scatter": true, "reduce_bucket_size": 2e8, "contiguous_gradients": true},
  "gradient_accumulation_steps": $ACCUM,
  "gradient_clipping": 0.5,
  "steps_per_print": 50,
  "train_micro_batch_size_per_gpu": $MICRO,
  "wall_clock_breakdown": false
}
EOF
echo "job=$SLURM_JOB_ID node=$(hostname) DATASET=$DATASET files=$N_FILES train=$N_TRAIN gpus=$NGPU micro=$MICRO accum=$ACCUM" \
     "global_batch=$((MICRO * NGPU * ACCUM)) steps/epoch=$STEPS_PER_EPOCH epochs=$EPOCHS total=$TOTAL lr=$LR warmup=$WARMUP" \
     "max_seq_len=${TRAIN_MAX_LEN:-from dir name}" \
     "target=$TARGET_DIR init=${INIT_DIR:-scratch} commit=$(git -C "$REPO" rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

cd "$REPO/eagle-train"
torchrun --standalone --nproc_per_node="$NGPU" eagle_trainer.py \
    --deepspeed_config "$CFG" \
    --base_model_path "$TARGET_DIR" \
    ${INIT_DIR:+--init_draft_path "$INIT_DIR"} \
    --data_path "$HS_ROOT" \
    --output_dir "$CKPT_DIR" \
    --project_name FastRL-offline-drafter \
    --experiment_name "eagle-$TARGET_TAG-$DATASET" \
    --batch_size "$MICRO" \
    --epochs "$EPOCHS" \
    --precision bf16 \
    --validate \
    ${TRAIN_MAX_LEN:+--max_seq_len "$TRAIN_MAX_LEN"}

python "$REPO/reproducibility/offline_drafter/export_drafter.py" --ckpt "$CKPT_DIR" --out "$EXPORT_DIR" \
    --target "$TARGET_DIR" --layout "$LAYOUT_DIR" ${INIT_DIR:+--init "$INIT_DIR"}
echo "drafter ready: $EXPORT_DIR"
