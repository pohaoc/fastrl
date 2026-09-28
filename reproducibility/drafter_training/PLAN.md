# Plan: TLT with opportunistic drafter training on a 4-engine rollout topology

Status: spike test on DAPO in progress (2026-09-28). Follows the 5-step frozen-drafter study in
[`../dataset/`](../dataset/README.md).

## Goal

Measure what TLT's *opportunistic drafter training* adds on top of speculation alone, and how much
idle straggler-tail GPU time it actually harvests. This is the TLT side of the SysX-vs-TLT comparison:
SysX spends idle GPUs on RL training, TLT spends them on continually training the drafter.

Questions to answer:

1. How many idle GPU-seconds does the rollout tail offer per RL step, and how many does TLT harvest
   for drafter training?
2. How much does continual drafter training raise acceptance (batch-average and straggler) and cut
   rollout / end-to-end time, relative to a frozen drafter?
3. Does drafter training ever slow the step down (contention with rollout or RL training)?

## Why the topology has to change

TLT only trains the drafter on a rollout worker that has finished its share of the batch and released
its memory while other workers are still generating (`verl/workers/rollout/sglang_rollout/worker_manager.py`,
`_check_and_start_training`, with `min_workers_for_training=1`). A "worker" is a data-parallel rollout
rank.

The 5-step study ran rollout as **one engine over all four GPUs** (`tensor_model_parallel_size=4`,
DP=1). That single worker only finishes when the whole rollout does, so there is never an idle worker
during the tail and TLT's drafter training can never start, even when enabled.

New topology: **four engines, one GPU each** (`actor_rollout_ref.rollout.tensor_model_parallel_size=1`,
DP=4). Each engine gets a quarter of the batch; engines finish at different times, and early finishers
can train the drafter while stragglers run. (Fallback if memory or speed forces it: 2 engines x TP=2.)

Consequences to keep in mind when reading results:

- Each engine sees about a quarter of the batch, so its running batch drops below SD's threshold (32)
  much earlier than one engine holding the whole batch: SD will be active for a larger share of rollout.
- The end-to-end straggler is whichever engine finishes last; the other three engines' tails become idle
  GPU time.
- TP=1 decodes a single request more slowly than TP=4, so tails are longer in wall-clock time.
- **None of the 5-step numbers can serve as this experiment's baseline.** SD-off and frozen-drafter runs
  must be re-run on the new topology.

## How drafter training runs (from the code)

- **Hand-off to the engines:** the drafter lives as an FSDP module on every worker; at each rollout
  wake-up, after the policy weights, its weights are pushed into every SGLang engine
  (`update_drafter_weights` in `verl/workers/sharding_manager/fsdp_sglang.py`). A drafter trained during
  step *k*'s tail is first used in step *k+1*.
- **Training data:** on the step before a training step, the actor's old-log-prob pass also returns the
  target model's hidden states (`collect_hidden_states_from_sgl=false`); the drafter trains on the
  *previous* step's rollouts.
- **Harvest is capped at `min_workers_for_training` GPUs.** Training starts once that many workers have
  released their memory; once a session is active, later-released workers do not join
  (`_check_and_start_training` returns early). With the default 1, at most one GPU trains; the other
  finished GPUs stay idle until the straggler engine ends. Sensitivity run: `min_workers_for_training` 2 and 3
  (more GPUs, later start). SysX, by contrast, can use every idle GPU.
- A training session runs at most 200 loop iterations; a successful optimizer step advances the counter
  twice (`_run_training_loop`), so a session does at most ~100 optimizer steps.

## Implementation status (2026-09-28)

- Launcher switches in `reproducibility/dataset/run_grpo_7B_4gpu.sh`: `ROLLOUT_TP` (default 4),
  `DRAFTER_TRAIN`, `DRAFTER_INTERVAL`, `DRAFTER_MIN_WORKERS`; defaults leave the 5-step study unchanged.
- Tracing: `worker_manager.py` emits `worker_released`, `worker_completed`, `drafter_train_session`
  (with `optimizer_steps`) and `drafter_train_step` spans (wall clock, no device sync).
- Outputs for this experiment: `OUT=reproducibility/drafter_training/outputs`.
- Not yet done: analysis of 4-engine runs (the existing `analyze.py` reads GPU 0's engine as *the* engine;
  with 4 engines the rollout ends at the last engine and each GPU has its own tail).

## Runs

All four on the same topology, model, data order (dataloader seed 1), hyperparameters and step count.

| Run | `speculative.enable` | `FASTRL_FORCE_PLAIN_DECODE` | `speculative.train.enable_drafter_training` | Isolates |
| --- | --- | --- | --- | --- |
| A. SD off (same-engine baseline) | true | 1 | false | reference |
| B. TLT, frozen drafter | true | unset | false | speculation alone |
| C. TLT, full | true | unset | true (interval 10, TLT default) | + continual drafter training |
| D. SysX | n/a | n/a | n/a | the method under study |

Optional sensitivity run: C with `training_interval_steps=1` (TLT's README warns that updating every step
can cause contention; this shows whether the default interval is leaving gains on the table).

Starting drafter for B and C: the released `mit-han-lab/Qwen2.5-7B-Eagle-RL` (trained on Eurus-style
prompts; acceptance at batch size 1 is 6.4–6.9 on Eurus, 5.65 on DAPO, 3.67 on SkyRL-SQL). The lower
out-of-domain acceptance is where continual training should help most — this favours TLT.

### Step count

- Drafter training happens once every `training_interval_steps=10` RL steps (data collected on the step
  before), and a frozen drafter only falls behind as the policy drifts. **Minimum 50 steps, target 100**
  (5–10 drafter updates).
- Enable checkpointing so an 8-hour run can resume: `trainer.save_freq=10` (keep 1 checkpoint; ~15 GB
  weights + optimizer state per checkpoint), `trainer.resume_mode=auto`.

### Datasets and order

1. **DAPO-Math** first (≈2 min/step at TP=4; cheapest, and drafter acceptance has room to improve).
2. **Eurus-2-RL** (≈5 min/step at TP=4; where SD already pays off).
3. **SkyRL-SQL**: TLT's drafter training is not reachable in multi-turn rollouts — it is only wired into
   the single-turn path (`_generate_with_drafter` -> `release_worker_memory`), and our
   `skyrl_env_rollout.py` path does not call it. Either (a) report SQL with run B only and state the
   limitation, or (b) add the bridge: after a worker's last trajectory finishes in
   `_skyrl_env_generate_sequences`, call `sharding_manager.release_memory()` and
   `drafter_manager.release_worker_memory()` per worker instead of once at the end. (b) needs its own
   validation run.

### Cost estimate (4x H100, per run, before the topology change's effect)

| Dataset | min/step (TP=4 today) | 50 steps | 100 steps |
| --- | --- | --- | --- |
| DAPO-Math | ~2 | ~1.7 h | ~3.5 h |
| Eurus-2-RL | ~5 | ~4 h | ~8.5 h |

Three TLT-side runs (A–C) per dataset. TP=1 changes per-step time; re-estimate after the smoke test.

## Configuration changes

Start from `reproducibility/dataset/run_grpo_7B_4gpu.sh`; add a `TOPO=dp4` switch and a drafter-training
switch rather than editing the existing settings (so the 5-step study stays reproducible):

```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=1   # 4 engines x 1 GPU (was 4)
speculative.train.enable_drafter_training=true            # run C only
speculative.train.training_interval_steps=10              # TLT default
speculative.train.min_workers_for_training=1              # TLT default
speculative.train.checkpoint_path=$OUT/<dataset>/drafter_ckpt
trainer.save_freq=10
trainer.resume_mode=auto
trainer.total_training_steps=100
```

Keep TLT's other drafter-training defaults as shipped (`batch_size_per_gpu=2`, `max_seq_len=8192`,
`max_epochs=10`, lr 1e-6, `collect_hidden_states_from_sgl=false`, i.e. drafter data = the target model's
hidden states from the actor's old-log-prob pass on the step before a training step).

Memory check: a 7B engine at TP=1 with `gpu_memory_utilization=0.4` takes ~38 GB of the 95 GB H100; the
drafter trainer adds the 1-layer drafter, its optimizer state and activations on an idle worker's GPU.

## Instrumentation to add

The existing tracing (see `../dataset/README.md#tracing`) covers rollout and RL training. Add:

- **Drafter-trainer spans** in `verl/workers/drafter/eagle_background_trainer.py`: one span per training
  session (start, stop, reason) and per optimizer step, tagged with GPU uuid, samples, loss, draft accuracy.
- **Worker state transitions** in `worker_manager.py`: GENERATING -> RELEASED -> TRAINING -> IDLE, with
  timestamps per worker. This gives, per GPU and step, the idle window and the part of it used for training.
- **Acceptance per RL step**: already in the `verify` spans (batch and per-request); keep.

## Analysis and figures

Extend `reproducibility/dataset/analyze.py`:

1. **Idle-time budget per step and GPU**: stacked bar of the straggler window split into
   *still generating* / *drafter training* / *idle* (the "how much tail time is repurposed" figure).
2. **Gain vs harvested time**: x = cumulative GPU-seconds of drafter training, y = rollout throughput or
   time saved vs run A, one line each for B and C.
3. **Acceptance over RL steps** for B vs C (batch-average and straggler), marking drafter-update steps.
4. **Step-time bars** (`step_bars.py`) for A, B, C (and D), and the SD cost breakdown (`sd_cost.py`) for B, C.

## Validation before the long runs

1. **Smoke test** on DAPO: TOPO=dp4, run C with `training_interval_steps=1`, 3 steps. Check that
   training sessions start on early-finishing workers (logs + new spans), stop cleanly before RL
   training, and that no step crashes (this path has never run on this machine).
2. Check drafter checkpoints are written and reloaded, and that acceptance in step 3 differs from a
   frozen-drafter run.
3. Re-estimate per-step time at TP=1 and finalize the step count.

## Deliverables

- `reproducibility/drafter_training/` with the launcher switches, a pair/quad runner, and a README
  mirroring `../dataset/README.md` (setup, baseline definition, results).
- Figures 1–4 above for DAPO and Eurus (and SQL per the decision above).
- Paper text: the TLT-comparison setup paragraph, updated with the final step count and topology.
