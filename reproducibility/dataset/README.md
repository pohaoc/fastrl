# Where does TLT's speculative-decoding speedup come from? Three-dataset reproduction

This folder reproduces our SD-on vs SD-off comparison of FastRL (TLT) GRPO training on three
datasets — **Eurus-2-RL**, **DAPO-Math-17k** and **SkyRL-SQL-653** — with per-GPU timeline traces and
a straggler-level analysis of where speculative decoding (SD) saves or loses time.

Everything runs from the fastrl repo root, on the code in this working tree (see
[Code changes this depends on](#code-changes-this-depends-on)).

## Quick start

```bash
source .venv/bin/activate                      # environment: see "Environment" below
bash   reproducibility/dataset/prepare_eurus.sh
python reproducibility/dataset/prepare_dapo.py
python reproducibility/dataset/prepare_sql.py  # downloads a 22 GB zip, keeps ~230 MB of databases

DATASET=eurus STEPS=5 bash reproducibility/dataset/run_pair.sh   # SD-on, SD-off, analysis
DATASET=dapo  STEPS=5 bash reproducibility/dataset/run_pair.sh
DATASET=sql   STEPS=5 bash reproducibility/dataset/run_pair.sh
```

`run_pair.sh` runs SD-on, then the SD-off baseline, checks that each finished all steps without a
traceback, and runs the analysis. Outputs land in `reproducibility/dataset/outputs/<dataset>/`:

| Path | Contents |
| --- | --- |
| `traces/sd_on/`, `traces/sd_off/` | JSONL timeline spans from every GPU process (see [Tracing](#tracing)) |
| `logs/` | Training and analysis logs |
| `results/gantt_step<N>.png`, `gantt_all_steps.png` | Per-GPU Gantt charts, SD off (top) vs SD on (bottom) |
| `results/critical_path.png` | Rollout time per step split into head, re-prefill and tail |
| `results/straggler_on.png` | The straggler's tokens per SD step against the break-even line |
| `results/summary.md`, `summary.json` | All tables below, per step |
| `results/step_bars.png` | Per-step wall time, speculative vs baseline, rollout (incl. reward) and training |
| `results/sd_cost.png`, `sd_cost.md` | Drafter vs verification cost per SD step and per training step |

Wall time for a pair on 4x H100: about 1 h for Eurus, 30 min for DAPO, 75 min for SQL.

A snapshot of our results is committed in [`results_4xH100/`](results_4xH100/) (charts and tables, no traces).

## Evaluation setup

### What is compared

For each dataset we run the same GRPO job twice with identical data order (dataloader seed 1),
model, hyperparameters and engine configuration:

- **SD on** — FastRL's adaptive EAGLE speculative decoding as shipped in `examples/grpo_7B.sh`:
  decode runs plain while more than `bs_threshold=32` requests are running; after 10 consecutive
  decode steps at or below 32 the engine clears its KV cache, re-prefills the running requests and
  speculates for the rest of that rollout. The draft config is chosen per batch size by the BEG
  bandit (`8_4_48` at batch size 1, down to `8_4_8` at 17–32).
- **SD off** — the controlled baseline described next.

Both runs use Qwen2.5-7B (base) with the released drafter `mit-han-lab/Qwen2.5-7B-Eagle-RL`,
4x H100 (95 GB), rollout tensor parallel 4 (one SGLang engine over all four GPUs), FSDP2 training with
Ulysses sequence parallel 4 and parameter/optimizer offload, 5 training steps, no validation, no
checkpoints.

### The SD-off baseline is NOT stock SGLang

Our baseline is **the same EAGLE engine with decode speculation forced off**, not FastRL with
`speculative.enable=false`. Setting `FASTRL_FORCE_PLAIN_DECODE=1` makes the EAGLE worker's
`should_enable_sd` return false for every decode batch (`third-party/sglang/.../eagle_worker.py`).
Everything else is identical to SD on:

| | SD on | SD off (ours) | Stock SGLang (`speculative.enable=false`) |
| --- | --- | --- | --- |
| Draft model loaded (same KV-cache capacity) | yes | yes | no |
| Overlap scheduler | off | off | on |
| `max_running_requests` / `cuda_graph_max_bs` | 512 / 32 | 512 / 32 | engine defaults |
| Prefill runs the drafter's `draft_extend` | yes | yes | no |
| Decode at batch > 32 | plain, no CUDA graph (graphs stop at 32) | same | plain, engine-default graphs |
| Decode at batch <= 32 | speculative after the switch | plain, CUDA-graphed | plain, CUDA-graphed |
| KV flush + re-prefill when SD turns on | yes | no | no |

**Why:** the question is *where SD's speedup comes from* — which steps, which batch sizes, which
requests. Against stock SGLang, a difference would mix speculation with unrelated engine changes:
the overlap scheduler (hides host time between steps), different CUDA-graph batch sizes and request
limits, and a larger KV cache because no drafter is resident. With our baseline the only difference
is whether decode speculates, so every second of difference is attributable to drafting,
verification, draft-extend and the SD switch, and each SD step can be compared with the plain decode
step the same engine would have run at the same batch size.

**What it does not measure:** the fixed cost of running an SD-capable engine at all (no overlap
scheduler, capped CUDA graphs, drafter memory), which stock SGLang avoids. Run `SD=stock` with
`run_grpo_7B_4gpu.sh` to measure that separately. The baseline also pays the drafter's prefill
(`draft_extend`), about 0.1 s per step in our runs.

### Datasets and settings

| | Eurus-2-RL | DAPO-Math-17k | SkyRL-SQL-653 |
| --- | --- | --- | --- |
| Source | `PRIME-RL/Eurus-2-RL-Data` | `BytedTsinghua-SIA/DAPO-Math-17k` (+ `AIME-2024` as validation file) | `NovaSky-AI/SkyRL-SQL-653-data-newfmt` + OmniSQL SQLite databases |
| Rows used | 480,537 | 17,398 (the published 1,791,700 rows repeat each prompt ~100x; deduplicated) | 653 (540 SynSQL, 113 Spider) |
| Prompt | Eurus action system prompt | Qwen default system prompt + DAPO instruction | SkyRL system template + schema/question (as published) |
| Rollout | single turn | single turn | multi-turn SkyRL-gym `text2sql` env: SQL tool, up to 5 turns |
| Reward | `prime_math` / `prime_code`: 1 / 0 | `math_dapo`: +1 / −1 from the `Answer:` line | SkyRL-gym `text2sql`: +1 match, 0 mismatch, −1 bad format |
| Prompts x samples per step | 64 x 8 | 64 x 8 | 256 x 5 |
| PPO mini-batch | 4 prompts | 4 prompts | 256 (one update per step) |
| Max prompt / response tokens | 1,024 / 32,768 | 1,024 / 32,768 | 4,096 / 11,264 |
| Temperature / top-p | 0.9 / 1.0 | 0.9 / 1.0 | 0.6 / 0.95 |
| KL loss / grad clip | 0.001 low-var-KL / 1.0 | 0.001 / 1.0 | none / 0.5 |
| LR | 1e-6 | 1e-6 | 1e-6 |

Eurus and DAPO use `examples/grpo_7B.sh` unchanged apart from 4 GPUs and 5 steps. SQL uses the
rollout and optimizer settings of granular-cais-rl's `sql_baseline.toml` (below).

### SkyRL-SQL: adapting the granular-cais-rl harness

The SQL setup follows the harness in `~/granular-cais-rl` (SkyRL-gym generator with
`use_conversation_multi_turn=false`). `verl/workers/rollout/sglang_rollout/skyrl_env_rollout.py`
reproduces its agent loop against FastRL's SGLang engine, enabled with
`+actor_rollout_ref.rollout.skyrl_env.enable=true`:

- The whole trajectory is **one assistant message**. Each turn generates up to 3,000 tokens until
  `</sql>` or `</solution>`; the environment's observation (query result or error, plus a turns-left
  reminder) is appended as plain tokens with loss mask 0; generation resumes, token-in-token-out.
- The loop ends when the env reports done (a `<solution>` or 5 turns) or when the context at the
  start of a turn exceeds 8,192 tokens. A trailing EOS is dropped between turns and appended at the
  end unless the last turn hit the length limit — as in the harness.
- The environment is **SkyRL-gym 0.3.0's `text2sql` env, unchanged** (the version the harness pins),
  including its reward. The reward is passed to the trainer as `rm_scores`, so it is never re-scored.
- Like the single-turn path, the rollout releases the engine's GPU memory before RL training
  (FastRL's sharding manager does not do this on exit).
- SGLang keeps the whole final token after a stop string; vLLM's `include_stop_str_in_output`
  (used by the harness) cuts the *text* right after it. We cut the text the same way before it
  reaches the env; token ids are kept as generated.

| Harness setting (`sql_baseline.toml`) | Here |
| --- | --- |
| Model Qwen2.5-Coder-7B-Instruct | **Qwen2.5-7B base** — the released drafter targets it (see below) |
| 256 prompts x 5, mini-batch 256 | same |
| 5 turns, 3,000 tokens/turn, 8,192 context, 4,096 prompt | same |
| Stop `</sql>`, `</solution>`, stop string kept | same |
| Temperature 0.6, top-p 0.95, top-k −1 | same |
| LR 1e-6 constant, grad clip 0.5, no KL, no entropy bonus, GRPO std-normalized | same |
| `drop_zero_signal=true` | not available; zero-variance groups get zero advantage but still count in the loss denominator |
| vLLM, TP 1 x 4 engines, `gpu_util` 0.9, eager | FastRL's SGLang, TP 4 x 1 engine, `gpu_memory_utilization` 0.4, CUDA graphs |
| granular `cpu_adam` + packed FSDP2 | verl FSDP2, Ulysses SP 4, dynamic batch, param/optimizer offload |

The last three rows are FastRL's own system and are identical for SD on and SD off, so they do not
affect this comparison — but they mean these runs do not measure FastRL against granular.

**Why Qwen2.5-7B base, not the harness's Coder model:** the only released TLT drafter was trained
on Qwen2.5-7B base. On 32 SQL prompts at batch size 1 it accepts **1.16** tokens per step with
Qwen2.5-Coder-7B-Instruct (85 tokens/s) against **3.67** with Qwen2.5-7B (268 tokens/s); break-even
is about 2.8. A Coder run would measure a mismatched drafter, not TLT. Using the base model keeps
the target and drafter matched and keeps all three datasets on the same model. The trade-off: the
base model often breaks the harness's format (e.g. it copies `</observation>` tags after an
observation), so most early rewards are −1. Reproduce the acceptance numbers with
`DATASET=sql bash bench_acceptance.sh` and `DATASET=sql MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct bash bench_acceptance.sh`.

### Tracing

Tracing is off unless `FASTRL_TRACE_DIR` is set; the launcher sets it.

- **SGLang** (`sglang/srt/fastrl_trace.py`): CUDA events around every scheduler forward pass and,
  inside the EAGLE worker, around `target_extend`, `draft_extend`, `decode_nosd`, `draft`, `verify`
  and `draft_extend_after_decode`. Events are resolved lazily (no extra synchronization) and mapped
  to wall-clock time through an anchor event re-taken whenever the scheduler goes idle. Each span
  records the batch size; `verify` spans also record per-request ids, accepted tokens, output
  lengths and finish flags, which is how the straggler is followed.
- **verl** (`verl/utils/timeline_trace.py`): wall-clock spans with device synchronization around
  `generate_sequences`, `compute_log_prob`, `compute_ref_log_prob` and `update_actor` on every GPU
  rank, plus the driver's phase timers (`step`, `gen`, `reward`, …).

Overhead is negligible: the Eurus benchmark ran at 494/464 tokens/s traced vs 464/462 untraced.

### Analysis definitions (`analyze.py`)

- **Head / tail:** the tail starts right after the last decode step with more than 32 running
  requests (SD's threshold). SD can only act in the tail.
- **Straggler:** the request still running at the last `verify` of the rollout with the longest
  output. For SQL each turn is a separate engine request, so this follows the straggler's last turn.
- **Break-even:** for an SD step at batch size *b*, SD beats plain decoding for the straggler only if
  it emits more than `period_SD / period_plain(b)` tokens for it, where `period_plain(b)` is the
  baseline's median step-to-step time at batch size *b* (host time included).
- **Time SD saved vs plain decode:** Σ over SD steps of `straggler_tokens x period_plain(b) −
  period_SD`, minus the re-prefill time. This compares SD with plain decoding of *the same tokens at
  the same batch sizes*, which is needed because the two runs' stragglers differ in length (SD
  changes which samples are drawn, not their distribution).
- Raw SD-on vs SD-off step times are also reported, but after step 1 the policies diverge, and even
  in step 1 the straggler lengths can differ, so they are not a clean measure of SD alone.

## Results (4x H100, 5 steps per run)

| | Eurus-2-RL | DAPO-Math-17k | SkyRL-SQL-653 |
| --- | --- | --- | --- |
| Drafter acceptance, 32 prompts, batch size 1 | 6.4–6.9 | 5.65 | 3.67 |
| Tail share of rollout (SD off) | 74–85% | 41–84% | 3–12% |
| Time SD saved vs plain decode, per step (s) | +40, −8, +102, +49, +29 | −30, +20, +6, +93, +3 | +0, −2, +3, −2, −1 |
| Straggler tokens per SD step, per step | 3.89, 2.66, 8.35, 6.41, 7.85 | 1.80, 5.31, 4.69, 8.94, 3.58 | 4.10, 3.28, 4.59, 3.48, 3.60 |
| Step-1 step time, SD off → on (s) | 395 → 320 | 250 → 194 | 421 → 430 |
| 5-step total, SD off → on (s), overall speedup | 1,656 → 1,166, 1.42x | 748 → 655, 1.14x | 1,994 → 1,986, 1.00x |

Findings:

- **All of SD's gain is in the tail**; head decoding, log-probs, actor update and weight sync take
  the same time in both runs.
- **The straggler's own acceptance decides the win, not the batch average.** Eurus step 2 and DAPO
  step 1 lost time: their stragglers averaged 2.66 and 1.80 tokens per SD step, at or below
  break-even, while the batch averages were above 4.
- **An SD step costs about 2.8 plain steps at batch size 1** (draft ≈ 47%, verify ≈ 48%, host ≈ 5%
  of the SD step), and the BEG tuner always uses its heaviest config (`8_4_48`) there, regardless of
  acceptance.
- **The largest wins come from capped stragglers accepting ~9 tokens per step** (Eurus steps 3–5,
  DAPO step 4), which looks like repetitive output running to the length cap. Generated text was
  not saved, so this is unconfirmed.

- **SkyRL-SQL gets no end-to-end speedup (1.00x overall, rollout 1.01x).** Its tail is only 3–12% of the
  rollout: 1,280 multi-turn trajectories keep the engine at its 512-request cap until the last 5–21 s.
  In that short tail, the SD switch's re-prefill (2.3–3.2 s per step, larger than on Eurus because
  multi-turn contexts are longer) cancels what speculation saves, and break-even is higher (3.0–3.3
  tokens) because most SD steps run at 17–32 requests, where verification is costlier. For SQL the
  straggler is its last turn's request, which hits the 3,000-token per-turn cap in every step.
- **Throughput and end-to-end time disagree.** Averaged over the batch, SD saves time in every step on all
  three datasets (`sd_cost.md`, `net_saved_s`); on the straggler, which bounds the step, it often does not.

## Environment

Tested on Ubuntu 24.04, 4x H100 (95 GB), driver with CUDA 13.4, Python 3.12.3 venv at `.venv`.
On top of the repo README's install (SGLang from `third-party/sglang`, flash-attn, `pip install -e .`):

- **torch 2.8.0 (cu128)** — install flash-attn's wheel with `--no-deps` or pip upgrades torch.
- **flashinfer 0.4.0** needs `apache-tvm-ffi==0.1.0b15`, which is no longer on PyPI, and `0.1.0` is
  API-incompatible. Build it from `apache/tvm-ffi` commit `7092774` (`pip wheel .`), then build
  flashinfer 0.4.0 from its sdist against that wheel.
- **CUDA 12.8 toolkit** for flashinfer's JIT kernels (the system had 13.4); installed with NVIDIA's
  runfile (`--toolkit --toolkitpath=$HOME/cuda-12.8`); export `CUDA_HOME` to it.
- `numpy<2` (verl), with `scipy==1.15.3` and `contourpy==1.3.3`.
- **SQL only:** `pip install skyrl-gym==0.3.0` (the harness's version).
- `verl/workers/rollout/sglang_rollout/sglang_rollout.py` creates an event loop when none exists
  (Python 3.12 + uvloop 0.22 raise otherwise).

## Code changes this depends on

All in this working tree; none change behaviour unless the env vars / config keys are set.

| File | Change |
| --- | --- |
| `third-party/sglang/python/sglang/srt/fastrl_trace.py` (new) | CUDA-event span tracer |
| `third-party/sglang/python/sglang/srt/managers/scheduler.py` | `forward` spans; flush traces when idle; plain decode preparation whenever adaptive SD is inactive (fixes a crash when a multi-turn running batch is rebuilt from a small prefill batch) |
| `third-party/sglang/python/sglang/srt/speculative/eagle_worker.py` | EAGLE sub-phase spans, per-request verify data; `FASTRL_FORCE_PLAIN_DECODE` |
| `verl/utils/timeline_trace.py` (new) | wall-clock spans |
| `verl/utils/profiler/performance.py` | driver/worker timers emit spans |
| `verl/workers/fsdp_workers.py` | per-GPU spans around rollout, log-prob and update methods |
| `verl/workers/rollout/sglang_rollout/skyrl_env_rollout.py` (new) | SkyRL-gym agent loop |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | `skyrl_env` rollout path; event-loop fix |

## Files

| File | Purpose |
| --- | --- |
| `prepare_eurus.sh`, `prepare_dapo.py`, `prepare_sql.py` | Download and convert each dataset to verl parquet |
| `run_grpo_7B_4gpu.sh` | One run: `SD=on\|off\|stock DATASET=eurus\|dapo\|sql STEPS=N` |
| `run_pair.sh` | SD-on + SD-off + checks + analysis for one dataset |
| `analyze.py` | Gantt charts, per-step breakdown, straggler and break-even analysis |
| `sd_cost.py` | Decomposes each SD step into the plain step it replaces + drafting + draft-extend + extra verification + extra host time |
| `step_bars.py` | Per-step speculative vs baseline wall-time bars |
| `results_4xH100/` | Committed snapshot of our results |
| `bench_acceptance.sh`, `bench_speculative_decoding.py` | Drafter acceptance on 32 prompts of a dataset |
