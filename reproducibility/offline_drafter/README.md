# Offline in-domain EAGLE drafters (DAPO-Math, SkyRL-SQL)

This fine-tunes the released drafter `mit-han-lab/Qwen2.5-7B-Eagle-RL` offline, on Qwen2.5-7B's own rollouts for
each dataset. It approximates the best that TLT's continual drafter training could reach, without running
online training, which is not usable yet (see `../notes/HANDOFF.md`).

- **Caveat: policy drift.** The data comes from the base policy (RL step 0), so the drafter matches the start
  of RL training. It does not follow the policy as RL moves it.
- **Caveat: more training than TLT gets.** Offline training sees far more data and epochs than TLT harvests
  during rollout tails, so this is an optimistic stand-in for TLT's drafter-quality lever.

## Pipeline (Oscar, L40S)

| Stage | Script | Environment | Resources | Output under `/oscar/scratch/$USER/eagle/` |
| --- | --- | --- | --- | --- |
| env | `build_env.sbatch` | builds `~/envs/eagle-train` (torch 2.6, transformers 4.51.1, DeepSpeed, flash-attn 2.7.4) | CPU | — |
| SQL data | `prepare_sql.sbatch` | fastrl | CPU | `$DATA_ROOT/SkyRL-SQL` (verl format + SQLite databases) |
| 1. rollouts | `gen.sbatch` → `gen_rollouts.py` | fastrl (SGLang, no speculation, fa3) | 1x L40S | `rollouts/<ds>/data/train.parquet` (pre-tokenized) |
| 2. hidden states | `datagen.sbatch` → `eagle-train/eagle_datagen.py` | eagle-train | 4x L40S | `hidden_states/<ds>/<ds>-<N>K/data_*.pt` |
| 3. fine-tune + export | `train.sbatch` → `eagle-train/eagle_trainer.py`, `export_drafter.py` | eagle-train | 4x L40S | `ckpt/<ds>/` (DeepSpeed), `drafters/Qwen2.5-7B-Eagle-RL-<ds>/` (HF format) |

To submit everything, with the stages chained by `afterok` dependencies, run from the repo root:
`bash reproducibility/offline_drafter/submit_pipeline.sh dapo sql`

## Data

The sampling settings match the RL runs (`../dataset/run_grpo_7B_4gpu.sh`).

**DAPO-Math**
- Prompts: 9,200 of the 17,398 deduplicated training prompts, chosen with seed 0.
- Held-out prompts: the other 8,198, written to `rollouts/dapo/heldout.parquet` for acceptance benchmarks.
- Sampling: temperature 0.9, top-p 1.0, one response per prompt, capped at 3,584 new tokens.
- Training sequences are at most 4K tokens.

**SkyRL-SQL**
- Prompts: all 653, with 4 trajectories each.
- Sampling: temperature 0.6, top-p 0.95, in the SkyRL-gym text2sql loop (5 turns, 3,000 tokens per turn,
  8,192-token context).
- Training sequences are at most 12K tokens.

**Loss mask.** Rows are pre-tokenized: they hold the engine's exact tokens plus a loss mask. The mask is 1 only
on tokens the model generated. For SQL, the environment observations are appended inside the same assistant
message, so a message-level mask would train the drafter on text it never drafts.

## Training

- Starts from the released drafter via `--init_draft_path`, a small addition to `eagle_trainer.py`.
- `fc` and `layers.0` are trained. `embed_tokens` and `lm_head` are frozen copies of the target's.
- DeepSpeed ZeRO-2 on 4 GPUs with torch AdamW, global batch 16.
- Learning rate: 2e-5 peak, 50 warmup steps, then linear decay.
- Epochs: DAPO 5, SQL 10, with validation after every epoch on eagle-train's 5% split.
- `EPOCHS`, `LR` and `WARMUP` override the defaults.

## Changes to eagle-train

- `eagle_datagen.py`: accepts pre-tokenized rows (`input_ids` and `loss_mask` columns).
- `eagle_trainer.py`: adds `--init_draft_path` (fine-tune from an EAGLE checkpoint) and `--validate`, which runs
  the existing `validate()` that was commented out.
- The learning rate, warmup and micro-batch come from the DeepSpeed JSON. The trainer's `--learning_rate` and
  `--warmup_steps` flags are unused, so `train.sbatch` writes the JSON.

## Evaluation

The exported directory works wherever the released drafter does:
- **Standalone acceptance:**
  `DATASET=dapo EAGLE_PATH=<export dir> PROMPTS_PARQUET=<rollouts/dapo/heldout.parquet> sbatch ../oscar/bench_acceptance.sbatch`
  Compare against the default drafter on the same prompts.
- **In the RL loop:** `SPEC_MODEL_PATH=<export dir>` with `../oscar/grpo.sbatch`, with SD on and off.
