# Oscar environment for FastRL jobs; sourced by the .sbatch files in this folder.
# Reuses the `fastrl` conda env, whose verl and sglang are editable installs of this checkout
# (/oscar/home/pch/fastrl and third-party/sglang/python).
module load cuda/12.9.0-cinr   # nvcc for sglang's JIT kernels
FASTRL_ENV=${FASTRL_ENV:-$HOME/.conda/envs/fastrl}
export PATH=$FASTRL_ENV/bin:$PATH

SCRATCH_ROOT=${SCRATCH_ROOT:-/oscar/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf}
export DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/fastrl}
# Run outputs (logs, checkpoints); home has a small quota.
export RUN_ROOT=${RUN_ROOT:-$SCRATCH_ROOT/fastrl-runs}

# L40S nodes are PCIe-only; P2P between the cards is not available.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

# The conda env's editable installs point at /oscar/home/pch/fastrl; make a job import THIS checkout
# (e.g. the sql-port worktree) instead. Python puts PYTHONPATH ahead of the .pth editable finders.
_FASTRL_CHECKOUT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH=$_FASTRL_CHECKOUT:$_FASTRL_CHECKOUT/third-party/sglang/python${PYTHONPATH:+:$PYTHONPATH}

# GPU health check for the start of a job: any GPU with uncorrectable ECC errors, or that cannot run a
# kernel, fails the job at once with a clear message. gpu3001 failed this way twice (2026-09-30, 2026-10-04),
# hence the --exclude in the sbatch files.
gpu_preflight() {
    nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total,retired_pages.pending --format=csv
    python - <<'PY' || { echo "GPU PREFLIGHT FAILED on $(hostname): exclude this node and resubmit"; exit 3; }
import torch
for i in range(torch.cuda.device_count()):
    x = torch.ones(1 << 20, device=f"cuda:{i}")
    assert float((x * 2).sum()) == 2.0 * (1 << 20), f"cuda:{i} computed a wrong result"
    torch.cuda.synchronize(i)
print(f"GPU preflight OK: {torch.cuda.device_count()} GPUs")
PY
}
