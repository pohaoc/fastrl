#!/bin/bash
# GRPO on SkyRL-SQL-653 with Qwen2.5-Coder-7B-Instruct on 4x L40S, following examples/grpo_7B.sh.
# Rollout = the SkyRL-gym text2sql agent loop of granular-cais-rl (use_conversation_multi_turn=False,
# see verl/workers/rollout/sglang_rollout/skyrl_env_rollout.py). Every parameter shared with the SysX SQL runs
# matches their synchronous configuration, ~/scratch/runs/fresh-sql-sync-8065792/run_config.toml ("SYNC" below).
# Deviations from examples/grpo_7B.sh are marked "# DEV:".
# Called by grpo_sql.sbatch / sql_pair.sbatch (env.sh already sourced). Env:
#   SD=true|false  SPEC_MODEL_PATH=<drafter dir>  STEPS  ATTN (fa3)  TP (1)  MEM (0.85)  OUT (run dir)  NAME
# Extra arguments are Hydra overrides.
set -euo pipefail
: "${SD:?}" "${OUT:?}" "${NAME:?}"
STEPS=${STEPS:-10}
ATTN=${ATTN:-fa3}
TP=${TP:-1}       # SYNC vllm_tp_size = 1: four single-GPU engines
MEM=${MEM:-0.85}  # SYNC vllm_gpu_util = 0.9 counts all engine memory; sglang's mem_fraction_static counts weights +
                  # KV only, so 0.85 leaves the rest (~10% of the GPU) for CUDA graphs and activations
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-Coder-7B-Instruct', revision='c03e6d358207e414f1eca0bb1891e29f1db0e242'))")
SPEC_MODEL_PATH=${SPEC_MODEL_PATH:-$(python -c "from huggingface_hub import snapshot_download as s; print(s('mit-han-lab/Qwen2.5-7B-Eagle-RL', revision='38905764986fdd43c64f0b5ab2a0bdac6bf71b08'))")}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export MKL_SERVICE_FORCE_INTEL=1
mkdir -p "$OUT"
echo "run=$NAME SD=$SD STEPS=$STEPS ATTN=$ATTN TP=$TP MEM=$MEM drafter=$SPEC_MODEL_PATH out=$OUT start=$(date) overrides: $*"

DATA_PATH=$DATA_ROOT/SkyRL-SQL                 # reproducibility/sql/prepare_sql.py
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
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=8192 \
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
    trainer.default_local_dir=$OUT/ckpt \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$STEPS \
    ray_init.num_cpus=${SLURM_CPUS_PER_TASK:?} \
    "$@"
# DEV (lines above): skyrl_env.* (SQL agent loop, SYNC environment settings; 2048 env workers need the raised
# thread limit in sql_pair.sbatch), chunked_prefill_size 8192 = SYNC vllm_max_num_batched_tokens, engine attention
# backend (verl forces fa3 when unset, so fa3 == upstream), n_gpus_per_node 8 -> 4, save_freq -1,
# total_training_steps, Oscar ray_init.num_cpus. Speculation settings (FastRL defaults) are TLT's own.
