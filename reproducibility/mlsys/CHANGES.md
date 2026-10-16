# Changes to TLT (FastRL)

Everything this branch changes relative to upstream FastRL, `mit-han-lab/fastrl` at commit `bce3df7`
("[Minor] Update README."). `git diff bce3df7 -- verl third-party eagle-train` shows the code; commits are on branch
`sql-port` of `pohaoc/fastrl`, which keeps the full history.

Our SQL runs used commit `19558c2`. This branch adds only changes that cannot affect them: step-log retraction events
(`630fc34`), the DAPO batch-preparation fix (`075453b`, which only acts when speculation runs with more than 32
requests, and the SQL arm never did: 88,609 speculative steps, all at 32 or fewer), and drafter-pipeline options
(`feba274`).

Each change says which arm it touches. "Both" means SD on and SD off alike; "SD on" means only the engine with
speculative decoding. Switches are environment variables; the launcher (`run_sql.sh`) sets the ones our runs used.

## 1. Correctness of speculative decoding

| Change | Arm | Where | Commit |
|---|---|---|---|
| **Restore the drafter's weights after every wake-up.** With 1.3 below, the drafter's weights live in a releasable memory region. `torch_memory_saver` serves all regions from one pool and labels memory per segment, so some drafter tensors land in segments of the target's `weights` region, which verl releases without a CPU copy and refills by pushing only the target's weights. Those drafter tensors came back as garbage: acceptance 1.02 instead of 3.91. Before the first release, the engine now keeps a pinned CPU copy of the drafter's own parameters and buffers (not the embedding and LM head it shares with the target) and copies it back in place after every resume, so CUDA graphs stay valid. | SD on | `sglang/srt/fastrl_sd_memory.py` (`snapshot_drafter`, `restore_drafter`), `managers/scheduler_update_weights_mixin.py` | `b5f0562` |
| **Serve the drafter as trained (zero `fc.bias`).** eagle-train builds the drafter's `fc` layer without a bias; sglang's `Qwen2ForCausalLMEagle` / `LlamaForCausalLMEagle` declare one and leave it uninitialised when the checkpoint has none. The export now writes a zero bias. The SQL drafter we provide already contains it. | SD on | `reproducibility/offline_drafter/export_drafter.py` | `60c0d20` |
| **Speculation switch: requeue the request in mid-prefill.** When adaptive SD switches on, FastRL clears the KV allocator and prefix cache and requeues running and waiting requests, but not the request in the middle of a chunked prefill; its stale KV slots then entered the new prefix cache. It is now requeued too, and every requeued request is reset as sglang resets a retracted one. `FASTRL_TRANSITION_FIX=0` restores upstream. | SD on | `managers/scheduler.py` (`event_loop_normal`) | `19558c2` |
| **Batch preparation follows whether the worker will speculate.** Upstream prepared every batch above the switch threshold (32) as a plain decode step, but once adaptive SD is on the EAGLE worker speculates at any batch size; a batch that grows back above 32 (DAPO, KV-bound) got a speculative step on plain-decode bookkeeping and every response became garbage. A plain step is now prepared only while SD is off. Never triggered in SQL. `FASTRL_PREP_FIX=0` restores upstream. | SD on | `managers/scheduler.py` (`update_running_batch`) | `075453b` |
| **Multi-turn rollouts: plain-decode preparation while SD is off.** In SkyRL-SQL every live trajectory can be between turns at once; the running batch is then rebuilt from a small prefill batch with spec algorithm EAGLE while adaptive SD is still off. It now gets plain-decode preparation, as large batches do. | SD on | `managers/scheduler.py` (`update_running_batch`, `sd_inactive`) | `f6e2532` |
| **Clip speculative overshoot past the length limit.** Speculative verification can accept a few tokens past `max_new_tokens`. Responses are cut to the limit, which is what decoding without speculation returns. | SD on (in practice) | `verl/.../sglang_rollout.py` (single-turn), `skyrl_env_rollout.py` (per SQL turn) | `333ba22` |

## 2. Memory: fitting TLT on 44 GB GPUs

| Change | Arm | Where | Commit |
|---|---|---|---|
| **Release the drafter's weights during training.** Upstream keeps the drafter (and every CUDA graph) resident while verl trains. The drafter's weights now go into a `weights_drafter` memory-saver region with a CPU backup, released and resumed with the target's weights (about 0.9 GB per GPU saved). `FASTRL_RELEASE_SD_MEMORY=0` restores upstream. | SD on | `fastrl_sd_memory.py`, `model_runner.py`, `scheduler_update_weights_mixin.py`, `constants.py` | `13074c5` |
| CUDA-graph release is **opt-in and off** (`FASTRL_RELEASE_SD_GRAPHS=1`): capturing graphs inside a memory-saver region broke the caching allocator. So, as upstream, the speculation CUDA graphs stay resident during training: about 3.6 GB per GPU for the SD engine against 0.8 GB without speculation. | SD on | `fastrl_sd_memory.py`, `cuda_graph_runner.py`, `eagle_draft_*_cuda_graph_runner.py` | `00b65ef` |
| **Wake-up order.** At the first step the engine still held its weights and KV cache when the actor's parameters were loaded for the weight push (the engine released itself only inside the first resume). The sharding manager now releases a freshly started engine first, and the launcher sets verl's `multi_stage_wake_up=True` (push weights, then re-allocate the KV cache). | Both | `verl/workers/sharding_manager/fsdp_sglang.py`; launcher | `941b80c` |
| **Expandable segments while training.** The SD arm's actor update ran out of memory at step 5 from fragmentation (4.7 GB reserved but unusable). verl's worker now turns on PyTorch's expandable segments for log-probabilities and the actor update and off before each rollout (the weight push shares tensors with the engine over CUDA IPC), as current upstream verl does (`set_expandable_segments` in `verl/workers/engine_workers.py`). `FASTRL_EXPANDABLE_SEGMENTS=1`. On our L40S this costs the SD arm about 60 s per actor update (it trains closer to the memory limit); see README. | Both | `verl/workers/fsdp_workers.py` | `19558c2` |
| **No unused rollout log-probabilities.** The single-turn rollout requested log-probabilities only to read back token ids; on re-prefilled requests sglang then computed fp32 logits over whole 8K-token chunks (about 4 GB). Requested only with `calculate_log_probs`; tokens come from `output_ids`. The SQL loop never requested them. | Both (single-turn path only) | `verl/.../sglang_rollout.py` | `a2e9f64` |
| **KV cache pinned** at 305,152 tokens per GPU in both arms (79% of the SysX run's), via `engine_kwargs.sglang.max_total_tokens`. Configuration, not code. | Both | launcher | `3426208` |

## 3. Robustness (no effect on results)

| Change | Arm | Where | Commit |
|---|---|---|---|
| Rollout sync timeout 600 s → 7200 s (`actor_rollout_ref.nccl_timeout`): with TP 1 every GPU waits for the slowest GPU's rollout. | Both | launcher | `88eb15d` |
| sglang's idle KV self-check logs instead of killing the engine (`FASTRL_KV_LEAK_WARN=1`); verl flushes the cache after every rollout. Never fired in SQL. | Both | `managers/scheduler.py` (`check_memory`) | `a2e9f64` |
| `FSDPEngineConfig` accepts `mixed_precision` (bf16 gradient reduction override). Not used by the SQL launcher. | Both | `verl/workers/config/engine.py` | `941b80c` |

## 4. Features needed for the workload

| Change | Arm | Where | Commit |
|---|---|---|---|
| **SkyRL-SQL agent loop.** Multi-turn rollouts through the SkyRL-gym `text2sql` environment, as SkyRL's generator runs them with `use_conversation_multi_turn=False` (the SysX setting): one assistant message per trajectory, stop at `</sql>` / `</solution>`, observations appended with loss mask 0, at most 5 turns of 3,000 tokens and a 24,000-token context. Enabled by `actor_rollout_ref.rollout.skyrl_env.enable=true`. | Both | `verl/.../skyrl_env_rollout.py`, `sglang_rollout.py` (`_skyrl_env_generate_sequences`) | `f6e2532` |
| **Engine settings pass-through.** Every non-null `actor_rollout_ref.rollout.engine_kwargs.sglang.*` entry (chunked prefill, `max_total_tokens`, ...) reaches the sglang engine; they override FastRL's speculative defaults where both set a key. | Both | `sglang_rollout.py` | `44c03ef` |
| SkyRL-SQL data preparation (prompts as published, OmniSQL SQLite databases). | — | `reproducibility/sql/prepare_sql.py` | `f6e2532` |

## 5. Instrumentation (no effect on results)

| Change | Where | Commit |
|---|---|---|
| Per-iteration engine log when `FASTRL_STEPLOG_DIR` is set: one JSON line per scheduler step (time, forward mode, batch size, queue, prefill tokens, CUDA graph, speculated, accepted tokens) plus `sd_transition`, `sd_reset` and `retract` events. One small write per iteration, in both arms. | `sglang/srt/fastrl_steplog.py`, `scheduler.py`, `eagle_worker.py` | `407464a`, `630fc34` |
| Ablation switch `FASTRL_FORCE_PLAIN_DECODE=1`: keep the EAGLE engine but never speculate on decode. Off by default. | `eagle_worker.py` | `407464a` |

## 6. Offline drafter training (how the provided drafter was made)

| Change | Where | Commit |
|---|---|---|
| Pipeline: rollouts from the target with the RL sampling settings → target hidden states → EAGLE fine-tune → export in FastRL's drafter format. Oscar/Slurm scripts, kept for provenance. | `reproducibility/offline_drafter/` | `f6e2532` and later |
| eagle-train: pre-tokenized rows with a loss mask (`eagle_datagen.py`); warm start from an EAGLE checkpoint (`--init_draft_path`) and per-epoch validation (`--validate`) in `eagle_trainer.py`. | `eagle-train/` | `f6e2532` |

The SQL drafter: Qwen2.5-Coder-7B-Instruct EAGLE head, warm-started from `mit-han-lab/Qwen2.5-7B-Eagle-RL` (fc and
layer 0 only), fine-tuned 10 epochs on 2,612 SkyRL-SQL trajectories (653 prompts × 4) sampled from the target at
T 0.6 / top-p 0.95 in the same agent loop, sequences truncated to 8K tokens; top-1 training accuracy 0.95.

## Unchanged

FastRL's speculation settings (EAGLE, 8 steps, top-k 4, 48 draft tokens, BEG over the MAB configs 8_4_32 / 8_4_16 /
8_4_8, switch on at ≤ 32 running requests, overlap scheduler off for the SD engine), its adaptive switch with
re-prefill, GRPO, the reward (SkyRL-gym `text2sql`, unchanged), and online drafter training (off in our runs, as in
upstream's default).
