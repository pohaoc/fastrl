#!/bin/bash
# One arm of the TLT (FastRL) SkyRL-SQL experiment: GRPO on SkyRL-SQL-653 with Qwen2.5-Coder-7B-Instruct, with or
# without FastRL's adaptive speculative decoding. Use sql_sd_on.sh / sql_sd_off.sh (they set SD and NAME).
# Settings are those of our runs (job 7333365, 4x L40S); every parameter shared with the SysX SQL runs matches their
# synchronous configuration ("SYNC" below). Deviations from upstream FastRL's examples/grpo_7B.sh are marked "# DEV:".
# All changes to FastRL itself are listed in CHANGES.md.
#
# Required env:
#   SD=true|false                 speculative decoding on / off
#   NAME                          run name (trainer.experiment_name)
#   OUT                           output directory (logs, per-iteration engine logs in $OUT/steplog)
#   DATA_PATH                     SkyRL-SQL directory from reproducibility/sql/prepare_sql.py (train.parquet,
#                                 validation.parquet, db/data)
#   SPEC_MODEL_PATH               drafter directory (SD=true only; config.json + pytorch_model.bin)
# Optional env:
#   MODEL_PATH                    target model directory (default: HF Qwen/Qwen2.5-Coder-7B-Instruct @ c03e6d3)
#   STEPS (35)  N_GPUS (4)  NUM_CPUS (nproc)  ATTN (fa3)  TP (1)  MEM (0.70)  KV_TOKENS (305152)
# Extra arguments are Hydra overrides.
set -euo pipefail
: "${SD:?}" "${OUT:?}" "${NAME:?}" "${DATA_PATH:?set DATA_PATH to the prepare_sql.py output}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Import verl and sglang from this checkout (works whether or not they are installed editable from it).
export PYTHONPATH=$REPO:$REPO/third-party/sglang/python${PYTHONPATH:+:$PYTHONPATH}
STEPS=${STEPS:-35}
N_GPUS=${N_GPUS:-4}
NUM_CPUS=${NUM_CPUS:-$(nproc)}
ATTN=${ATTN:-fa3}
TP=${TP:-1}       # SYNC vllm_tp_size = 1: one single-GPU engine per GPU
# DEV: KV cache pinned at 305,152 tokens per GPU in both arms (79% of SYNC's 387,184; Qwen2.5-7B: 57,344 B per
# token), passed to sglang as max_total_tokens. On 44 GB L40S this was the largest pool both arms fit (the colocated
# verl worker keeps ~2.7-4.4 GB per GPU while generating; the SD engine needs ~5.5 GB more than SD off). Keep it on
# larger GPUs so the KV cache, and with it batching and retraction behaviour, matches our runs. mem_fraction_static
# only has to cover weights + the pinned pool; at 0.70 the pin binds in both arms.
KV_TOKENS=${KV_TOKENS:-305152}
MEM=${MEM:-0.70}
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-Coder-7B-Instruct', revision='c03e6d358207e414f1eca0bb1891e29f1db0e242'))")
fi
if [ "$SD" = true ]; then
    : "${SPEC_MODEL_PATH:?SD=true needs SPEC_MODEL_PATH (the drafter directory)}"
else
    SPEC_MODEL_PATH=${SPEC_MODEL_PATH:-none}   # unused when SD=false
fi
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export MKL_SERVICE_FORCE_INTEL=1
export PYTHONUNBUFFERED=1          # print verl's per-step metrics as they happen, not only at exit
export FASTRL_KV_LEAK_WARN=1        # sglang KV leak check logs instead of killing the engine (CHANGES.md)
export FASTRL_EXPANDABLE_SEGMENTS=1 # expandable segments while the verl worker trains (CHANGES.md)
export FASTRL_STEPLOG_DIR=${FASTRL_STEPLOG_DIR:-$OUT/steplog}   # per-iteration engine log (analyze_pair.py)
# 2048 SkyRL env workers (the SYNC setting) exceed the default soft thread limit with Ray/SGLang/torch threads.
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
mkdir -p "$OUT" "$FASTRL_STEPLOG_DIR"
echo "run=$NAME SD=$SD STEPS=$STEPS N_GPUS=$N_GPUS ATTN=$ATTN TP=$TP MEM=$MEM KV_TOKENS=$KV_TOKENS model=$MODEL_PATH drafter=$SPEC_MODEL_PATH data=$DATA_PATH out=$OUT commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo ?) start=$(date) overrides: $*"
cd "$REPO"

# DEV: dataset/rollout/optimizer = SYNC: 256 prompts x 5, one update per step (mini = batch), lr 1e-6, weight
# decay 0.01 (verl default), grad clip 0.5, constant LR without warmup, no KL loss, no entropy term, no zero-signal
# filter, std-normalized GRPO advantages, T 0.6 / top-p 0.95 / top-k -1, <=5 turns of <=3000 tokens, 4096-token
# prompts, 24000-token context limit at the start of a turn, seed 42.
train_prompt_bsz=256
n_resp_per_prompt=5
train_prompt_mini_bsz=256
max_prompt_length=4096
max_input_length=24000
# Sequence cap = SYNC max_input_length. 5 turns x 3000 generated tokens + prompt + observations stays below it in
# practice; a longer response would be truncated (skyrl_env_rollout logs a warning if that ever happens).
max_response_length=$((max_input_length - max_prompt_length))
SP=4
# DEV: per-GPU token budget x SP = 24000 = SYNC token_budget per micro-batch. verl also requires budget x SP >= the
# sequence cap, which holds. Fused kernels keep full-vocab logits out of memory (L40S, 44 GB). Micro-batching only.
ppo_max_token_len_per_gpu=$(( max_input_length / SP ))

python3 -m verl.trainer.main_fastrl \
    speculative.eagle.spec_model_path=$SPEC_MODEL_PATH \
    speculative.enable=$SD \
    speculative.bs_threshold=32 \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/validation.parquet \
    data.return_raw_chat=True \
    data.return_full_prompt=True \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.seed=42 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=$MEM \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=8192 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=$KV_TOKENS \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=$ATTN \
    +actor_rollout_ref.rollout.skyrl_env.enable=true \
    +actor_rollout_ref.rollout.skyrl_env.max_turns=5 \
    +actor_rollout_ref.rollout.skyrl_env.max_generate_length=3000 \
    +actor_rollout_ref.rollout.skyrl_env.max_input_length=$max_input_length \
    "+actor_rollout_ref.rollout.skyrl_env.stop=['</sql>','</solution>']" \
    +actor_rollout_ref.rollout.skyrl_env.max_env_workers=2048 \
    +actor_rollout_ref.rollout.skyrl_env.env_configs.text2sql.db_path=$DATA_PATH/db/data \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=FastRL-SQL \
    trainer.experiment_name=$NAME \
    trainer.default_local_dir=$OUT/ckpt-$NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$STEPS \
    ray_init.num_cpus=$NUM_CPUS \
    "$@"
# DEV (lines above): skyrl_env.* (SQL agent loop, SYNC environment settings; 2048 env workers need the raised
# thread limit set above), chunked_prefill_size 8192 = SYNC vllm_max_num_batched_tokens, engine attention
# backend (verl forces fa3 when unset, so fa3 == upstream), n_gpus_per_node 8 -> 4, save_freq -1,
# total_training_steps, ray_init.num_cpus. Speculation settings (FastRL defaults) are TLT's own.
# multi_stage_wake_up: push the new weights before re-allocating the KV cache, so the actor's parameters and the
# KV cache are never on the GPU together (without it both SQL arms ran out of memory in the first wake-up, job 7227378).
# nccl_timeout 600 -> 7200 s: with TP 1 every GPU waits at a barrier for the slowest GPU's rollout, and one SD-arm
# GPU finished >10 min after the others (gloo timeout, job 7283745).
