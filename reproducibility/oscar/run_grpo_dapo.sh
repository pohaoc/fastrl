#!/bin/bash
# GRPO on DAPO-Math-17k with DeepSeek-R1-Distill-Llama-8B on 4x H100, following examples/grpo_7B.sh.
# Every parameter shared with the SysX DAPO runs matches their synchronous configuration,
# ~/scratch/runs/dapo-b128-sync-8065792/run_config.toml ("SYNC" below). Deviations from examples/grpo_7B.sh are
# marked "# DEV:".
# Called by dapo_pair.sbatch (env.sh already sourced). Env:
#   SD=true|false  SPEC_MODEL_PATH=<drafter dir>  STEPS  ATTN (fa3)  TP (1)  MEM (0.92)  OUT (run dir)  NAME
# Extra arguments are Hydra overrides.
set -euo pipefail
: "${SD:?}" "${OUT:?}" "${NAME:?}"
STEPS=${STEPS:-20}   # SysX DAPO sync ran max_steps = 10; we run 20 (the first 10 compare directly)
ATTN=${ATTN:-fa3}
TP=${TP:-1}         # SYNC vllm_tp_size = 1: four single-GPU engines
# DEV: KV cache pinned at 72.5% of SYNC's, the same in both arms. SYNC pins vllm_kv_cache_memory_bytes =
# 59,223,572,480 B = 451,840 tokens of Llama-8B KV (2 x 32 layers x 8 KV heads x 128 x 2 B = 131,072 B per token),
# but weights (15.0 GiB) + that KV (55.2 GiB) leave ~7 GiB of an H100's 79.2 GiB, too little for CUDA graphs, prefill
# activations and the ~3 GiB the colocated verl worker keeps: at mem 0.92 the SD arm ran out of memory capturing the
# drafter's graphs and the SD-off arm in its first prefill (job 7283229); at 360,448 tokens the SD arm still ran out
# in a prefill (3.87 GiB logits buffer, 3.77 GiB free; job 7306292). 327,680 tokens = 40.0 GiB, passed to sglang as
# max_total_tokens; mem_fraction_static 0.80 (63.3 GiB) is above weights + drafter + target and draft KV (~57 GiB) in
# either arm, so the pin sets the pool, and leaves ~20 GiB for graphs, activations and the worker.
KV_TOKENS=${KV_TOKENS:-327680}
MEM=${MEM:-0.80}
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download as s; print(s('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', revision='6a6f4aa4197940add57724a7707d069478df56b1'))")
SPEC_MODEL_PATH=${SPEC_MODEL_PATH:-${SCRATCH_ROOT:-/oscar/scratch/$USER}/eagle/DeepSeek-R1-Distill-Llama-8B/drafters/DeepSeek-R1-Distill-Llama-8B-Eagle-dapo}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export MKL_SERVICE_FORCE_INTEL=1
export FASTRL_KV_LEAK_WARN=1   # sglang KV leak check logs instead of killing the engine (see scheduler.check_memory)
export FASTRL_EXPANDABLE_SEGMENTS=1  # expandable segments while the verl worker trains (fsdp_workers._fastrl_expandable_segments)
mkdir -p "$OUT"
echo "run=$NAME SD=$SD STEPS=$STEPS ATTN=$ATTN TP=$TP MEM=$MEM KV_TOKENS=$KV_TOKENS drafter=$SPEC_MODEL_PATH out=$OUT start=$(date) overrides: $*"

DATA_PATH=/oscar/data/deeptir/mborjigi/datasets/dapo_math17k   # SYNC train_data / val_data
# DEV: dataset/rollout/optimizer = SYNC: 128 prompts x 8, one update per step (mini = batch), lr 1e-6, weight
# decay 0.01 (verl default), grad clip 0.5, constant LR without warmup, no KL loss, no entropy term, no zero-signal
# filter, std-normalized GRPO advantages, T 1.0 / top-p 1.0 / top-k -1, single turn, prompts <= 1024 tokens,
# responses <= 32000 tokens (context 33024 = SYNC max_input_length), seed 42, bf16 gradient reduction, and the
# SkyRL-gym "aime" reward (reproducibility/dapo/aime_reward.py).
train_prompt_bsz=128
n_resp_per_prompt=8
train_prompt_mini_bsz=128
max_prompt_length=1024
max_response_length=32000
SP=4
# DEV: per-GPU token budget x SP = 35000 = SYNC token_budget per micro-batch (>= the 33024-token sequence cap, as
# verl requires). Fused kernels keep full-vocab logits out of memory. Micro-batching only.
ppo_max_token_len_per_gpu=$(( 35000 / SP ))

python3 -m verl.trainer.main_fastrl \
    speculative.eagle.spec_model_path=$SPEC_MODEL_PATH \
    speculative.enable=$SD \
    speculative.bs_threshold=32 \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/val.parquet \
    data.return_raw_chat=True \
    data.return_full_prompt=True \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.seed=42 \
    custom_reward_function.path=$(pwd)/reproducibility/dapo/aime_reward.py \
    custom_reward_function.name=compute_score \
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
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=$MEM \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=8192 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=256 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=$KV_TOKENS \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=$ATTN \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=FastRL-DAPO \
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
# DEV (lines above): custom reward (SYNC env_class aime), engine settings = SYNC vllm_max_num_batched_tokens (8192),
# vllm_max_num_seqs (256), vllm_kv_cache_memory_bytes (as max_total_tokens) via engine_kwargs.sglang.*, engine
# attention backend (verl forces fa3 when unset, so fa3 == upstream), bf16 reduce = SYNC reduce_dtype,
# n_gpus_per_node 8 -> 4, save_freq -1, total_training_steps, Oscar ray_init.num_cpus. Speculation settings
# (FastRL defaults) are TLT's own. Training memory strategy (param + optimizer offload) is verl's equivalent of
# SYNC's trainer offload during rollout + cpu_adam: the pinned 59 GB KV cache needs the GPU while generating.
# multi_stage_wake_up: push the new weights before re-allocating the KV cache, so the actor's parameters and the
# KV cache are never on the GPU together (without it both SQL arms ran out of memory in the first wake-up, job 7227378).
# nccl_timeout 600 -> 7200 s: with TP 1 every GPU waits at a barrier for the slowest GPU's rollout, and one SD-arm
# GPU finished >10 min after the others (gloo timeout, job 7283745).
