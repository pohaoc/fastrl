#!/bin/bash
# SGLang attention backends for FastRL's EAGLE engine on 4x B200 (fa3 kernels are Hopper-only): each backend with
# speculation on (sd) and off (plain = same engine, FASTRL_FORCE_PLAIN_DECODE=1), batch size 1, 2 Eurus prompts
# generated to 16K tokens. One process per configuration, so a backend that fails does not stop the others.
# Submit from the repo root: sbatch reproducibility/oscar/bench_backends.sbatch   [BACKENDS="triton fa4 flashinfer"]
#SBATCH --job-name=fastrl-bench-backends
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --constraint=b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -uo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
source "$REPO/reproducibility/oscar/env.sh"
export HF_HUB_OFFLINE=1
BACKENDS=${BACKENDS:-"triton fa4 flashinfer"}
OUT=$RUN_ROOT/bench_backends/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git -C "$REPO" rev-parse HEAD) backends=$BACKENDS out=$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
cd "$REPO"
for b in $BACKENDS; do
    for m in plain sd; do
        echo "===== $b $m $(date)"
        timeout 1500 python reproducibility/oscar/bench_backends.py --backend "$b" --mode "$m" \
            --trace_dir "$OUT/$b-$m" 2>&1 | grep -v -E "hf_quant_config|OfflineMode|retry\(\)|Capturing batches|UserWarning|warnings.warn" | tail -25
        echo "===== $b $m rc=${PIPESTATUS[0]}"
    done
done
python reproducibility/oscar/bench_backends.py --summarize "$OUT"
