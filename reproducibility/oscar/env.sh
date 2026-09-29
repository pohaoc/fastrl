# Oscar environment for FastRL jobs; sourced by the .sbatch files in this folder.
# Reuses the `fastrl` conda env, whose verl and sglang are editable installs of this checkout.
module load cuda/12.9.0-cinr
FASTRL_ENV=${FASTRL_ENV:-$HOME/.conda/envs/fastrl}
export PATH=$FASTRL_ENV/bin:$PATH

SCRATCH_ROOT=${SCRATCH_ROOT:-/oscar/scratch/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf}
# Datasets in the layout run_grpo_7B_4gpu.sh expects (built by prepare_data.sbatch).
export DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data/fastrl}
# Run outputs (traces, checkpoints); home has a small quota.
export RUN_ROOT=${RUN_ROOT:-$SCRATCH_ROOT/fastrl-runs}
