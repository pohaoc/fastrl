# TLT on SkyRL-SQL: speculative decoding on vs off

This folder reproduces our TLT (FastRL) baseline runs on SkyRL-SQL: GRPO on SkyRL-SQL-653 with
Qwen2.5-Coder-7B-Instruct, once with FastRL's adaptive speculative decoding (EAGLE, offline-tuned drafter) and once
without, with every other setting identical. Everything we changed in FastRL is listed in [CHANGES.md](CHANGES.md).

| Script | What it runs |
|---|---|
| `sql_sd_on.sh` | the SD-on arm: FastRL with adaptive EAGLE speculative decoding |
| `sql_sd_off.sh` | the SD-off arm: FastRL with speculation disabled (plain sglang engine, overlap scheduler on) |
| `run_sql.sh` | the shared launcher both call; all training and rollout settings live here |
| `analyze_pair.py` | compares the two arms (phase times, rollout times, acceptance vs break-even, time to score) |

The DAPO experiment is not included yet.

## 1. Environment

Install FastRL as upstream does (Python 3.12, CUDA 12.8), from **this checkout**:

```bash
git clone -b mlsys-repro https://github.com/pohaoc/fastrl.git && cd fastrl
conda create -n fastrl python=3.12 -y && conda activate fastrl
pip install -e "third-party/sglang/python[all]"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install -e .
pip install skyrl-gym==0.3.0          # the SkyRL-SQL environment and reward
```

Versions in our environment, for reference: torch 2.8.0, sglang 0.5.3.post2 (this checkout's `third-party/sglang`),
sgl-kernel 0.3.15, torch_memory_saver 0.0.9rc2, transformers 4.57.1, flash_attn 2.8.3, flashinfer-python 0.4.0,
ray 2.58.0, tensordict 0.9.1, skyrl-gym 0.3.0, numpy 1.26.4. `run_sql.sh` puts this checkout on `PYTHONPATH`, so
it runs this code even if verl or sglang were installed from elsewhere.

The attention backend is FlashAttention 3 (`ATTN=fa3`, the FastRL default), which runs on Hopper (H100/H200).

## 2. Data

```bash
python reproducibility/sql/prepare_sql.py --out /path/to/SkyRL-SQL
```

This downloads NovaSky-AI/SkyRL-SQL-653-data-newfmt and the OmniSQL database bundle (22 GB zip; only the SQLite
databases the data references are extracted) and writes `train.parquet`, `validation.parquet` and `db/data/`.
Pass that directory as `DATA_PATH`.

## 3. Models

- **Target:** `Qwen/Qwen2.5-Coder-7B-Instruct`, revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`. `run_sql.sh`
  downloads it unless `MODEL_PATH` points to a local copy.
- **Drafter** (SD on only; sent separately): directory `Qwen2.5-Coder-7B-Instruct-Eagle-sql/` with `config.json`,
  `pytorch_model.bin` and `export_info.json`. Pass it as `SPEC_MODEL_PATH`. Check it is the one we used:

  ```
  sha256  a6b2f70c41bb98aa0550de03a79ff4aada19e20cff7f0dd2b6f5aecf646543c7  config.json
  sha256  e0aae488065a1c24b1f7b3bb6ec6ece672fe959e907d5c5820053e8ca79a6025  pytorch_model.bin
  ```

  How it was trained is in [CHANGES.md](CHANGES.md#6-offline-drafter-training-how-the-provided-drafter-was-made).

## 4. Run

One node with 4 GPUs per arm. Run the arms one after the other on the same node, so node speed does not differ
between them:

```bash
export DATA_PATH=/path/to/SkyRL-SQL OUT=/path/to/runs/sql-pair
SPEC_MODEL_PATH=/path/to/Qwen2.5-Coder-7B-Instruct-Eagle-sql bash reproducibility/mlsys/sql_sd_on.sh
bash reproducibility/mlsys/sql_sd_off.sh
python reproducibility/mlsys/analyze_pair.py $OUT summary.json
```

Each arm runs 35 RL steps (`STEPS`) and writes `$OUT/sd{on,off}.log` (verl's per-step metrics) and
`$OUT/sd{on,off}/steplog/` (one JSON line per engine iteration per GPU). No checkpoints are saved. Other knobs:
`N_GPUS` (4), `NUM_CPUS` (all cores, for Ray), `MODEL_PATH`; extra arguments are passed to verl as Hydra overrides.

**Settings** (in `run_sql.sh`; every parameter shared with the SysX SQL run matches its synchronous configuration):
256 prompts × 5 samples, one update per step, lr 1e-6, grad clip 0.5, no KL, GRPO; T 0.6 / top-p 0.95; at most 5
turns of 3,000 tokens and a 24,000-token context; prompts up to 4,096 tokens; seed 42; TP 1 (one engine per GPU);
chunked prefill 8,192; KV cache pinned at 305,152 tokens per GPU; FSDP2 with parameter and optimizer offload,
Ulysses sequence parallelism 4, 6,000 tokens per GPU per micro-batch.

## 5. What we measured (4× L40S 44 GB, job 7333365, commit 19558c2)

| Average per step, 35 steps | SD on | SD off |
|---|---|---|
| Generation (incl. engine wake-up and weight sync) | 187.7 s | 243.7 s |
| Log-probabilities | 203.4 s | 198.5 s |
| Actor update | 743.0 s | 681.1 s |
| **Step** | **1,134.9 s** | **1,123.7 s** |
| Total, 35 steps | 11.03 h | 10.92 h |

- Rollout time from the engine logs: 6,201 s with SD vs 8,245 s without (1.33×).
- Acceptance: median 5.21 tokens per request per verify step; break-even median 2.47 (speculation only runs at 32 or
  fewer running requests).
- Learning is unchanged: training score 0.5 at step 12 in both arms (4.22 h / 4.20 h), 0.65 at step 22 (7.40 h /
  7.29 h).
- The SD arm's actor update is about 62 s slower per step. On our 44 GB L40S, the paused SD engine keeps its
  speculation CUDA graphs on the GPU during training (about 3.6 GB against 0.8 GB), and with expandable segments on
  that leaves the trainer near the memory limit, where remapping memory stalls. On 141 GB H200s this gap should
  shrink or disappear; that difference is expected, and worth reporting.

## 6. Running on H200 (or other GPUs)

- Keep **4 GPUs per arm** (`N_GPUS=4`) and **TP 1**: the batch splits into four engines of 320 trajectories each,
  which sets batching, retraction and tail behaviour.
- Keep **`KV_TOKENS=305152`**: with more memory sglang would size a larger KV pool and the rollouts would batch
  differently from ours. `MEM=0.70` only has to cover weights plus the pinned pool, which it does on any larger GPU.
- Keep `FASTRL_EXPANDABLE_SEGMENTS=1` (set in `run_sql.sh`) so the code path matches ours, even though H200s would
  not need it to avoid running out of memory.
- Absolute times will differ from ours (faster GPUs, NVLink). The comparison is SD on vs SD off on the same node.
- On PCIe-only nodes (like our L40S) set `NCCL_P2P_DISABLE=1`; not needed with NVLink.
