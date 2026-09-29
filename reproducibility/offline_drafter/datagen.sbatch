#!/bin/bash
# Stage 2: cache Qwen2.5-7B last-layer hidden states for the rollouts (eagle-train/eagle_datagen.py, EAGLE-2 mode).
# ~7 KB per token on disk (bf16 x 3584): DAPO ~40 GB, SQL ~70-140 GB, under $EAGLE_ROOT/hidden_states/.
# Submit from the repo root: DATASET=dapo|sql sbatch reproducibility/offline_drafter/datagen.sbatch
#SBATCH --job-name=eagle-datagen
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
source "$REPO/reproducibility/offline_drafter/paths.sh"
source "$EAGLE_ENV/bin/activate"
export HF_HUB_OFFLINE=1
NGPU=$(nvidia-smi -L | wc -l)
TARGET_DIR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$TARGET_MODEL'))")
echo "job=$SLURM_JOB_ID node=$(hostname) DATASET=$DATASET gpus=$NGPU in=$ROLLOUTS out=$HS_DIR target=$TARGET_DIR"
mkdir -p "$HS_DIR"
cd "$REPO/eagle-train"
torchrun --standalone --nproc_per_node="$NGPU" eagle_datagen.py \
    model.base_model_path="$TARGET_DIR" \
    data.data_path="$ROLLOUTS" \
    data.save_dir="$HS_DIR" \
    data.max_length=$((MAX_K * 1024)) \
    data.sample_ratio=1.0 \
    data.mode=eagle2 \
    hydra.run.dir="$EAGLE_ROOT/hydra/$DATASET-$SLURM_JOB_ID"   # not under $HS_ROOT: the trainer walks that tree
echo "files: $(ls "$HS_DIR" | grep -c '^data_.*\.pt$')"; cat "$HS_DIR/stats.json"; du -sh "$HS_DIR"
