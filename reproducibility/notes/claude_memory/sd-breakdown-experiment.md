---
name: sd-breakdown-experiment
description: SD-on vs SD-off GRPO-7B timeline experiment (2026-09-28): where it lives, how the baseline works, key findings
metadata:
  type: project
---
Experiment in `/home/cc/fastrl/experiments/sd_breakdown/` (uncommitted): `run_grpo_7B_4gpu.sh` (SD=on|off|stock, STEPS), `analyze.py`, traces/, results/.
Tracing is enabled by env `FASTRL_TRACE_DIR` (sglang `srt/fastrl_trace.py` CUDA-event spans; verl `verl/utils/timeline_trace.py`).
SD=off is a controlled baseline: same SD engine, `FASTRL_FORCE_PLAIN_DECODE=1` makes `should_enable_sd` false on decode (bs_threshold=0 would break scheduler prepare_for_decode).

Findings (5 steps, 4xH100): step 1 (clean comparison) 1.23x step / 1.35x rollout / 1.47x tail. Gain is all in the bs<=32 tail.
Straggler's own acceptance differs from batch mean (step1 3.89 vs 5.49; step2 2.66 vs 5.25, net ~0 vs plain decode). bs=1 always uses 8_4_48 (~11ms/step vs 3.9ms plain, break-even ~2.8 tok).
**Why:** user cares about straggler critical path, not batch-average acceptance ([[fastrl-env-setup]]).
**How to apply:** judge SD by straggler tokens/step vs break-even; runs diverge after step 1, so compare later steps per straggler token or via counterfactual.

DAPO-Math (2026-09-28, `DATASET=dapo`, `run_pair.sh`, results in results/dapo/): data deduped to 17,398 train prompts; drafter bench accept 5.65 vs Eurus 6.4-6.9.
Step 1 stragglers differ even from identical weights (off 32,768 vs on 12,826 tokens) because SD changes the samples. SD net vs plain decode per step: -30, +20, +6, +93, +3 s; step-1 straggler averaged 1.80 tokens/SD step (below ~2.8 break-even for 79 s); step 4's +93 s came from a capped straggler accepting ~9/step (likely repetition). Excluding step 4, SD roughly broke even.

Checkpoint (2026-09-29): branch `sd-breakdown-repro` in /home/cc/fastrl (commits e42ef4c, b135d74, d814910 and later); remote `fork` = https://github.com/pohaoc/fastrl.git (user's fork; push needs the user's credentials, none stored on this box). Artifact lives in reproducibility/dataset/ (README with evaluation setup + SD-off baseline rationale) and reproducibility/drafter_training/PLAN.md (next: 4-engine TP=1 rollout + TLT opportunistic drafter training, runs A-D, >=50-100 steps, DAPO first). SkyRL-SQL uses Qwen2.5-7B base (user choice, released drafter matches it).

SkyRL-SQL results (2026-09-28): 1.00x overall (1,994 -> 1,986 s), rollout 1.01x; tail only 3-12% of rollout; re-prefill 2.3-3.2 s/step cancels SD's gain. Fixed a scheduler bug (plain-decode prep skipped when a multi-turn running batch is rebuilt from a small EAGLE prefill batch) (commit b135d74). The user pushes from their own session (git identity pohaoc <pch@brown.edu>).

Drafter-training experiment (2026-09-29): launcher switches ROLLOUT_TP / DRAFTER_TRAIN / DRAFTER_INTERVAL / DRAFTER_MIN_WORKERS in reproducibility/dataset/run_grpo_7B_4gpu.sh; worker_manager.py emits drafter_train_* spans. DAPO spike (4 x TP=1 engines, interval 1, 3 steps) started; in step 1 worker 0 activated drafter training and cleaned up after ~2 s (no data yet, as expected). See reproducibility/notes/HANDOFF.md.
