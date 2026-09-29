# Handoff: where the SD breakdown and drafter-training work stands (2026-09-29)

Branch `sd-breakdown-repro` on https://github.com/pohaoc/fastrl. Read this first when resuming.

## Done

- **SD-on vs SD-off, 5 steps, 4x H100, Qwen2.5-7B + `mit-han-lab/Qwen2.5-7B-Eagle-RL`**, on Eurus-2-RL,
  DAPO-Math and SkyRL-SQL. Everything to reproduce it is in [`../dataset/`](../dataset/README.md)
  (setup, baseline rationale, results, committed result snapshot in `results_4xH100/`).
  Overall speedup: Eurus 1.42x, DAPO 1.14x, SQL 1.00x.
- **Paper-facing conclusions:** SD's gain is entirely in the batch<=32 tail; the straggler's own acceptance,
  not the batch average, decides the end-to-end win; an SD step costs ~2.7 plain steps at batch 1, about
  75% of the overhead is the drafter (8 sequential passes + draft-extend); SQL has almost no tail
  (oversubscribed engine, 3,000-token turns) and GPU idle time from tool waits is only ~1–2% of its tail.

## In progress: TLT opportunistic drafter training ([`../drafter_training/PLAN.md`](../drafter_training/PLAN.md))

Code is ready and committed:

- `reproducibility/dataset/run_grpo_7B_4gpu.sh` switches: `ROLLOUT_TP=1` (4 one-GPU engines),
  `DRAFTER_TRAIN=1`, `DRAFTER_INTERVAL` (TLT default 10), `DRAFTER_MIN_WORKERS` (default 1). Defaults are
  unchanged, so the 5-step study still reproduces.
- `verl/workers/rollout/sglang_rollout/worker_manager.py` emits `worker_released`, `worker_completed`,
  `drafter_train_session` (with `optimizer_steps`) and `drafter_train_step` spans into `FASTRL_TRACE_DIR`.

**Spike test (not finished when the machine was lost; rerun it first):**

```bash
source .venv/bin/activate
OUT=$PWD/reproducibility/drafter_training/outputs
SD=on DATASET=dapo STEPS=3 ROLLOUT_TP=1 DRAFTER_TRAIN=1 DRAFTER_INTERVAL=1 OUT=$OUT \
  FASTRL_TRACE_DIR=$OUT/dapo/traces/spike_sd_on_drafter \
  bash reproducibility/dataset/run_grpo_7B_4gpu.sh > $OUT/dapo/spike.log 2>&1
```

What was observed before it ended (step 1 of 3):

- Startup to the first rollout took ~20 min (4 engines, CUDA-graph capture, drafter FSDP init).
- All four engines registered (`dp=0..3`); GPUs at 85–91% during rollout.
- In step 1, each engine started a drafter-training session on its own GPU as it finished, in order
  workers 0, 2, 3, 1 (00:00:57 to 00:01:37), and each cleaned up ~2 s later. Expected: step 1 has no drafter
  data yet (it is collected during each step's old-log-prob pass). Note: the "one training GPU" limit only
  holds while a session is active; once a session ends, the next released worker can start a new one.

- **The run then died at the end of step 1 with host-RAM exhaustion** (Ray: node at 975 / 1,007 GB, worker
  killed) inside the old-log-prob pass that returns hidden states for the drafter
  (`batch.meta_info["return_hidden_states"]`, `ray_trainer.py` ~line 1215); the drafter buffer never got data.
  Hypothesis (unverified): `dp_actor.py` re-pads the last-layer hidden states to (batch, seq_len, hidden), i.e.
  512 x ~33K x 3584 x 2 bytes ≈ 120 GB per worker for DAPO, and they are returned through Ray to the driver and
  sent back to every worker (`add_drafter_data_to_buffer`). **Fix this before the rerun**: keep hidden states
  unpadded (only valid tokens) and on the worker that computed them, or subsample sequences / cap
  `max_seq_len` (drafter config default 8192) before returning them. Measure host RAM during old-log-prob.

Check after the rerun:

1. Steps 2–3 run `drafter_train_session` spans with `optimizer_steps > 0`, and no traceback.
2. The drafter checkpoint appears under `$OUT/dapo/drafter_ckpt/`.
3. Acceptance in step 3 differs from a frozen-drafter run at the same topology.
4. Per-step time at TP=1, to finalize the step count (plan: >= 50, target 100).

Then: extend `analyze.py` for 4 engines (rollout ends at the last engine; each GPU has its own tail), and run
A (SD off), B (frozen drafter), C (full TLT) on DAPO, then Eurus.

## Environment (rebuild notes)

See [`claude_memory/fastrl-env-setup.md`](claude_memory/fastrl-env-setup.md) and the Environment section of
[`../dataset/README.md`](../dataset/README.md): CUDA 12.8 toolkit, flashinfer 0.4.0 built against
apache-tvm-ffi 0.1.0b15 from commit `7092774`, flash-attn with `--no-deps`, numpy<2, skyrl-gym 0.3.0 for SQL.
Datasets are not in git: rebuild with `reproducibility/dataset/prepare_{eurus.sh,dapo.py,sql.py}`.

## Files in this folder

- `claude_memory/`: the assistant's session notes (context on decisions and findings).
- `../dataset/dev/`: one-off development scripts (SQL agent-loop test, run chaining, acceptance benchmark
  wrapper).
