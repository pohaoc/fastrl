#!/bin/bash
# Re-export a trained drafter from its latest DeepSpeed checkpoint (no GPU), e.g. after an export_drafter.py fix.
#   TARGET=r1-distill-7b DATASET=dapo sbatch reproducibility/offline_drafter/export.sbatch   (from the repo root)
# Same command as the end of train.sbatch; overwrites $EXPORT_DIR.
#SBATCH --job-name=eagle-export
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
source "$REPO/reproducibility/offline_drafter/paths.sh"
source "$EAGLE_ENV/bin/activate"
TARGET_DIR=$(snapshot "$TARGET_MODEL" "$TARGET_REVISION")
INIT_DIR=; [ "$INIT_DRAFT" = none ] || INIT_DIR=$(snapshot "$INIT_DRAFT" "$INIT_REVISION")
LAYOUT_DIR=$(snapshot "$LAYOUT_REF" "$LAYOUT_REVISION")
export HF_HUB_OFFLINE=1
echo "job=$SLURM_JOB_ID TARGET=$TARGET DATASET=$DATASET ckpt=$CKPT_DIR/$(cat "$CKPT_DIR/latest") out=$EXPORT_DIR commit=$(git -C "$REPO" rev-parse --short HEAD)"
python "$REPO/reproducibility/offline_drafter/export_drafter.py" --ckpt "$CKPT_DIR" --out "$EXPORT_DIR" \
    --target "$TARGET_DIR" ${LAYOUT_DIR:+--layout "$LAYOUT_DIR"} ${INIT_DIR:+--init "$INIT_DIR"}
python -c "
import torch; sd = torch.load('$EXPORT_DIR/pytorch_model.bin', map_location='cpu', weights_only=True)
print('fc tensors:', {k: (tuple(v.shape), float(v.abs().max())) for k, v in sd.items() if k.startswith('fc')})"
