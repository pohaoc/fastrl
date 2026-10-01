#!/bin/bash
# One run of reproducibility/dataset/run_grpo_7B_4gpu.sh on 4x H100 (gpu-he).
# Submit from the repo root. The run is configured by the launcher's env vars, which sbatch exports:
#   SD=on DATASET=dapo STEPS=1 ROLLOUT_TP=1 DRAFTER_TRAIN=0 sbatch reproducibility/oscar/grpo.sbatch
# Optional: RUN_GROUP (default drafter_training) picks OUT=$RUN_ROOT/$RUN_GROUP; extra sbatch
# arguments after the script name are passed to the launcher as Hydra overrides.
# Writes the job log to /oscar/scratch/$USER/jobs/, plus host-RAM and GPU samples and (on failure)
# Ray's logs next to the traces, because Slurm deletes the node's /tmp at job end.
#SBATCH --job-name=fastrl-grpo
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
LAUNCHER=$REPO/reproducibility/dataset/run_grpo_7B_4gpu.sh
[ -f "$LAUNCHER" ] || { echo "submit from the repo root (or set FASTRL_REPO)"; exit 1; }
source "$REPO/reproducibility/oscar/env.sh"
: "${SD:?set SD=on, off or stock}" "${DATASET:?set DATASET=eurus, dapo or sql}"

# Weights come from the HF cache in $HF_HOME; don't let a run depend on the network.
export HF_HUB_OFFLINE=1
export OUT=${OUT:-$RUN_ROOT/${RUN_GROUP:-drafter_training}}
# Ray: size to the allocation (not the 112-core node), own temp dir, and never `ray stop` other jobs.
export RAY_NUM_CPUS=$SLURM_CPUS_PER_TASK
export RAY_STOP=0
export RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID

RUN_TAG=sd_${SD}-tp${ROLLOUT_TP:-4}-drafter${DRAFTER_TRAIN:-0}-${SLURM_JOB_ID}
export FASTRL_TRACE_DIR=${FASTRL_TRACE_DIR:-$OUT/$DATASET/traces/$RUN_TAG}
MON_DIR=$OUT/$DATASET/monitor/$RUN_TAG
mkdir -p "$FASTRL_TRACE_DIR" "$MON_DIR" "$RAY_TMPDIR"

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) repo=$REPO"
echo "commit=$(git -C "$REPO" rev-parse HEAD)"; git -C "$REPO" status --short
echo "SD=$SD DATASET=$DATASET STEPS=${STEPS:-5} ROLLOUT_TP=${ROLLOUT_TP:-4} DRAFTER_TRAIN=${DRAFTER_TRAIN:-0}" \
     "DRAFTER_INTERVAL=${DRAFTER_INTERVAL:-10} OUT=$OUT FASTRL_TRACE_DIR=$FASTRL_TRACE_DIR overrides: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Host RAM every 30 s: "epoch job_GB node_used_GB" (job = cgroup v2 of the whole job, 0 if not
# found; the node is shared). The drafter spike peaked at 521 GB.
CG=/sys/fs/cgroup$(sed -n 's#^0::\(.*/job_[0-9]*\).*#\1#p' /proc/self/cgroup)
( while sleep 30; do
      echo "$(date +%s) $(( $(cat "$CG/memory.current" 2>/dev/null || echo 0) / 1073741824 ))" \
           "$(free -g | awk '/^Mem/{print $3}')"
  done ) > "$MON_DIR/host_mem_gb.log" &
MEM_PID=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 \
    > "$MON_DIR/gpu.csv" &
GPU_PID=$!

set +e
bash "$LAUNCHER" "$@"
RC=$?
set -e
kill $MEM_PID $GPU_PID 2>/dev/null || true
echo "launcher rc=$RC end=$(date) peak_host_mem_gb=$(sort -k2 -n "$MON_DIR/host_mem_gb.log" | tail -1 | cut -d' ' -f2)"
# Ray puts its session under $RAY_TMPDIR/ray/ (older layouts: directly under $RAY_TMPDIR). The worker logs hold
# the SGLang engine/scheduler output, which the job log does not.
for S in "$RAY_TMPDIR/ray/session_latest" "$RAY_TMPDIR/session_latest"; do
    if [ $RC -ne 0 ] && [ -d "$S/logs" ]; then
        cp -r "$S/logs" "$MON_DIR/ray_logs"
        echo "ray logs copied to $MON_DIR/ray_logs (read worker-*.err from the head)"
        break
    fi
done
exit $RC
