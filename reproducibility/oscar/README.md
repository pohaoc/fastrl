# Running on Oscar (Brown CCV)

The artifact was built on a 4x H100 (95 GB) machine with a `.venv`. On Oscar, the same launcher
(`../dataset/run_grpo_7B_4gpu.sh`) runs inside a Slurm job on 4x H100 from the `gpu-he` partition.

| File | Purpose |
| --- | --- |
| `env.sh` | Loads CUDA, puts the `fastrl` conda env on `PATH`, and sets `HF_HOME`, `DATA_ROOT` and `RUN_ROOT` on scratch |
| `prepare_data.sbatch` | One-off job (CPU). Builds `$DATA_ROOT/DAPO-Math-17k` (deduplicated) and links `$DATA_ROOT/Eurus-2-RL-Data` to the full Eurus data |
| `grpo.sbatch` | One launcher run on `--gres=gpu:h100:4`. Also logs host RAM and GPU samples, and keeps Ray's logs if the run fails |

## Oscar-specific settings (and why)

- **`ray_init.num_cpus` = the allocation** (`RAY_NUM_CPUS`). Without it, Ray sizes itself to the node's
  112 cores and its workers die at startup. The driver then hangs silently while the GPUs sit idle.
- **`RAY_STOP=0`**: the launcher's `ray stop --force` would kill other Ray jobs of the same user on a
  shared node. Each job also gets its own `RAY_TMPDIR`.
- **`--gres=gpu:h100:4`**: `gpu-he` mixes H100, L40S, B200 and RTX Pro nodes, so the GPU type must be
  named. L40S is not usable for SD-on runs yet, because the in-loop EAGLE draft graph crashes with fa3.
- **`--mem=700G`**: the 4 x TP=1 drafter-training spike peaked at 521 GB host RAM.
- **Attention backend**: verl's default (`fa3`). flashinfer's JIT kernels are broken in the conda env
  (its tvm-ffi pin), so never set `attention_backend=flashinfer`.
- **Offline HF**: models load from the cache in `$HF_HOME`, and jobs set `HF_HUB_OFFLINE=1`.
- **H100 memory**: Oscar's H100s may have less memory than the 95 GB cards used to build the artifact.
  If the actor update OOMs, append `actor_rollout_ref.model.use_fused_kernels=True`.

## Usage (from the repo root)

```bash
sbatch reproducibility/oscar/prepare_data.sbatch          # once
# First run on Oscar: the frozen-drafter diagnostic from ../notes/HANDOFF.md (1 step, 4 x TP=1)
SD=on DATASET=dapo STEPS=1 ROLLOUT_TP=1 DRAFTER_TRAIN=0 sbatch reproducibility/oscar/grpo.sbatch
```

Outputs go to `$RUN_ROOT/$RUN_GROUP/<dataset>/` (default `RUN_GROUP=drafter_training`):
- `traces/<tag>/`
- `monitor/<tag>/`, which holds `host_mem_gb.log`, `gpu.csv` and, after a failure, `ray_logs/`
- the job log, in `/oscar/scratch/$USER/jobs/`

For runs longer than 2 h, pass `--time` to sbatch. Put the launcher's Hydra overrides after the script name.
