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

- **Root cause found and fixed (2026-09-29, commit after a70d5da):**
  - With FastRL's shipped default `collect_hidden_states_from_sgl=false`, **the drafter never trains**:
    `EagleBackgroundTrainer._training_step_impl` returns early, and `DataBuffer` is created with
    `store_hidden_states=collect_hidden_states_from_sgl`, so the actor-side hidden states are discarded.
  - The OOM came from that discarded path: `dp_actor.compute_log_prob(return_hidden_states=True)` pads every
    sample back to the full sequence length (~242 MB per DAPO sample, ~124 GB per worker because sequence
    parallelism 4 gives each worker the whole batch), then ships them through the driver to every worker.
  - Fix: `DRAFTER_TRAIN=1` now sets `collect_hidden_states_from_sgl=true` (the drafter trains on hidden states
    the SGLang engine returns during rollout, via `collect_online_data`, which feeds both the cross-step buffer
    and the current-step store), and the actor-side collection in `ray_trainer.py` is now opt-in
    (`speculative.train.collect_hidden_states_from_actor`, default false).
  - **Unverified:** memory of engine-side collection (SGLang returns per-token hidden states for each engine's
    share of the batch). Watch host RAM during the first rollout of the rerun.

- **Rerun after the fix (spike2, DAPO, 4 x TP=1, interval 1): the drafter now trains.** Step 1 took 901 s
  (rollout 823 s, old-log-prob 19 s, ref 11 s, update 47 s); host RAM peaked at 386 / 1,007 GB. Timeline from
  step start (s): engines finished generating at 349 (w1), 419 (w2), 482 (w3), 622 (w0, the straggler).
  After generating, each worker spent 98–208 s collecting the engine's hidden states before marking itself
  complete (w0: 622 -> 811), which lands on the critical path for the straggler (~21% of the step).
  Drafter sessions: w1 365–506 s (64 optimizer steps, hit the per-session cap), w0 627–772 s (51 steps).
  w2 and w3 stayed idle after finishing (only one session at a time; no new session started when w1's
  ended, because sessions start only on a release event), ~730 GPU-seconds unused.
  For reference, DAPO step 1 with one 4-GPU engine took 194 s (SD on, no drafter training) / 250 s (SD off).
  The rollout itself is much slower at TP=1 (straggler engine done at 622 s vs a 105 s rollout at TP=4);
  a frozen-drafter TP=1 run is needed to separate topology from drafter-training cost.
- **Step 2 finished (678 s; rollout 615 s), but SD accepted nothing in either step:** mean 1.00 token per SD step
  (bonus token only) on all four engines, over 29,008 (step 1) and 24,195 (step 2) SD steps. The same drafter
  averaged 4–6 tokens per step at TP=4 without drafter training. Speculation was pure overhead, which likely
  explains most of the slow TP=1 rollout. Host RAM peaked at 521 GB. Drafter sessions in step 2: one, on w0
  (112 s, 100 optimizer steps).
  **Top suspect:** with drafter training on, every wake-up pushes the trainer's FSDP drafter module into the
  engines (`update_drafter_weights` in `fsdp_sglang.py`, called from `wake_up`), including before any training
  in step 1, and acceptance is already 1.00 in step 1. So the pushed weights are probably wrong (drafter module
  not loaded from `spec_model_path`, key names from `convert_weight_keys` not matching the engine's draft model,
  or dtype). **First thing to do next:** run 1 step with `ROLLOUT_TP=1 DRAFTER_TRAIN=0` (frozen drafter, no push).
  If acceptance is normal (4–6), the push is the bug: compare the engine's draft weights before and after the
  push, and check how the FSDP drafter module is initialized. If acceptance is also 1.00, SD at TP=1 is broken.
- Open issues for the full runs: (1) hidden-state collection time on the straggler (optimize or overlap it);
  (2) idle released workers never join or restart training; (3) interval 1 was for the spike only: use TLT's
  default 10 (collection runs only on the step before a training step).

Check after the rerun:

1. Steps 2–3 run `drafter_train_session` spans with `optimizer_steps > 0`, and no traceback.
2. The drafter checkpoint appears under `$OUT/dapo/drafter_ckpt/`.
3. Acceptance in step 3 differs from a frozen-drafter run at the same topology.
4. Per-step time at TP=1, to finalize the step count (plan: >= 50, target 100).

Then: extend `analyze.py` for 4 engines (rollout ends at the last engine; each GPU has its own tail), and run
A (SD off), B (frozen drafter), C (full TLT) on DAPO, then Eurus.

## Environment (rebuild notes)

**Resuming on Oscar (2026-09-28):** use [`../oscar/`](../oscar/README.md). It reuses the `fastrl` conda env
(editable installs of this checkout), 4x H100 on `gpu-he`, data and outputs on scratch. The first Oscar job is
the frozen-drafter diagnostic above (`SD=on DATASET=dapo STEPS=1 ROLLOUT_TP=1 DRAFTER_TRAIN=0`).

See [`claude_memory/fastrl-env-setup.md`](claude_memory/fastrl-env-setup.md) and the Environment section of
[`../dataset/README.md`](../dataset/README.md): CUDA 12.8 toolkit, flashinfer 0.4.0 built against
apache-tvm-ffi 0.1.0b15 from commit `7092774`, flash-attn with `--no-deps`, numpy<2, skyrl-gym 0.3.0 for SQL.
Datasets are not in git: rebuild with `reproducibility/dataset/prepare_{eurus.sh,dapo.py,sql.py}`.

## Files in this folder

- `claude_memory/`: the assistant's session notes (context on decisions and findings).
- `../dataset/dev/`: one-off development scripts (SQL agent-loop test, run chaining, acceptance benchmark
  wrapper).
