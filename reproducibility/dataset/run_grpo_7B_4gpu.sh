#!/bin/bash
# GRPO on Qwen2.5-7B, 4x H100, with timeline tracing; one run of one dataset and one SD mode.
# Usage (from anywhere): SD=on|off|stock DATASET=eurus|dapo|sql STEPS=5 \
#     bash reproducibility/dataset/run_grpo_7B_4gpu.sh
#   SD:
#     on    - adaptive speculative decoding, as in examples/grpo_7B.sh
#     off   - controlled baseline (used in our analysis): the SAME engine as `on` (EAGLE drafter
#             loaded, same scheduler, CUDA graphs and limits) with decode speculation forced off
#             via FASTRL_FORCE_PLAIN_DECODE=1. See README.md.
#     stock - speculative.enable=false: stock SGLang settings. Not used in our analysis.
#   DATASET:
#     eurus - Eurus-2-RL-Data (prepare_eurus.sh); settings = examples/grpo_7B.sh
#     dapo  - DAPO-Math-17k deduplicated + AIME-2024 validation (prepare_dapo.py); same settings
#     sql   - SkyRL-SQL-653 in the SkyRL-gym text2sql multi-turn env (prepare_sql.py);
#             rollout and optimizer settings from granular-cais-rl sql_baseline.toml
#   Optional (drafter-training experiment, see ../drafter_training/PLAN.md):
#     ROLLOUT_TP=1        rollout engines of 1 GPU each (4 engines); default 4 = one 4-GPU engine
#     DRAFTER_TRAIN=1     TLT opportunistic drafter training on released rollout workers
#     DRAFTER_INTERVAL=10 train every N RL steps (TLT default 10); DRAFTER_MIN_WORKERS=1 (TLT default)
#   Optional (cluster use, see ../oscar/README.md):
#     DATA_ROOT=<dir>     directory holding Eurus-2-RL-Data/, DAPO-Math-17k/, SkyRL-SQL/ (default: repo root)
#     RAY_NUM_CPUS=<n>    ray_init.num_cpus; required under Slurm, where Ray otherwise sizes itself to the node
#     RAY_STOP=0          skip `ray stop --force` (it would kill other Ray jobs of the same user on a shared node)
#   Extra arguments are appended as Hydra overrides.
# Differences from examples/grpo_7B.sh: 4 GPUs, total_training_steps=$STEPS, no checkpoints,
# traces written to $FASTRL_TRACE_DIR.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
SD=${SD:?set SD=on, off or stock}
DATASET=${DATASET:?set DATASET=eurus, dapo or sql}
STEPS=${STEPS:-5}
OUT=${OUT:-$REPO/reproducibility/dataset/outputs}
DATA_ROOT=${DATA_ROOT:-$REPO}
case $SD in
  on) SPEC_ENABLE=true ;;
  off) SPEC_ENABLE=true; export FASTRL_FORCE_PLAIN_DECODE=1 ;;
  stock) SPEC_ENABLE=false ;;
  *) echo "SD must be on, off or stock"; exit 1 ;;
esac
case $DATASET in
  eurus) DATA_PATH=$DATA_ROOT/Eurus-2-RL-Data ;;
  dapo) DATA_PATH=$DATA_ROOT/DAPO-Math-17k ;;
  sql) DATA_PATH=$DATA_ROOT/SkyRL-SQL ;;
  *) echo "DATASET must be eurus, dapo or sql"; exit 1 ;;
esac
export FASTRL_TRACE_DIR=${FASTRL_TRACE_DIR:-$OUT/$DATASET/traces/sd_$SD}
mkdir -p "$FASTRL_TRACE_DIR"

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export MKL_SERVICE_FORCE_INTEL=1

CKPT_PATH=${CKPT_PATH:-$OUT/ckpt}
PROJECT_NAME=FastRL
ROLLOUT_TP=${ROLLOUT_TP:-4}
DRAFTER_TRAIN=${DRAFTER_TRAIN:-0}
DRAFTER_INTERVAL=${DRAFTER_INTERVAL:-10}
DRAFTER_MIN_WORKERS=${DRAFTER_MIN_WORKERS:-1}
EXPERIMENT_NAME=Qwen2.5-7B-4gpu-${DATASET}-sd_${SD}
if [ "$ROLLOUT_TP" != 4 ] || [ "$DRAFTER_TRAIN" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}-tp${ROLLOUT_TP}-drafter${DRAFTER_TRAIN}
fi
MODEL_PATH=Qwen/Qwen2.5-7B
SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL

# Defaults = examples/grpo_7B.sh (Eurus, DAPO).
train_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=4
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 32))
temperature=0.9
top_p=1.0
use_kl_loss=True
grad_clip=1.0
total_epochs=1
EXTRA_ARGS=()
if [ "$DATASET" = sql ]; then
    # granular-cais-rl sql_baseline.toml: 256 prompts x 5, one update per step, no KL loss,
    # grad clip 0.5, temp 0.6 / top-p 0.95, <=5 turns, 3000 tokens per turn, 8192-token context
    # limit at the start of a turn (so a response can reach 8192 + 3000 tokens minus the prompt).
    train_prompt_bsz=256
    n_resp_per_prompt=5
    train_prompt_mini_bsz=256
    max_prompt_length=4096
    max_response_length=$((1024 * 11))
    temperature=0.6
    top_p=0.95
    use_kl_loss=False
    grad_clip=0.5
    total_epochs=100  # 653 prompts = 2 steps per epoch; total_training_steps caps the run
    EXTRA_ARGS=(
        +actor_rollout_ref.rollout.skyrl_env.enable=true
        +actor_rollout_ref.rollout.skyrl_env.max_turns=5
        +actor_rollout_ref.rollout.skyrl_env.max_generate_length=3000
        +actor_rollout_ref.rollout.skyrl_env.max_input_length=8192
        "+actor_rollout_ref.rollout.skyrl_env.stop=['</sql>','</solution>']"
        +actor_rollout_ref.rollout.skyrl_env.max_env_workers=2048
        +actor_rollout_ref.rollout.skyrl_env.env_configs.text2sql.db_path=$DATA_PATH/db/data
    )
fi
if [ "$DRAFTER_TRAIN" = 1 ]; then
    EXTRA_ARGS+=(
        speculative.train.enable_drafter_training=true
        speculative.train.training_interval_steps=${DRAFTER_INTERVAL}
        speculative.train.min_workers_for_training=${DRAFTER_MIN_WORKERS}
        # The drafter only trains on hidden states returned by the SGLang engine during rollout;
        # with the shipped default (false) its training step is skipped.
        speculative.train.collect_hidden_states_from_sgl=true
        speculative.train.checkpoint_path=$OUT/$DATASET/drafter_ckpt/$EXPERIMENT_NAME
    )
fi
if [ -n "${RAY_NUM_CPUS:-}" ]; then
    EXTRA_ARGS+=(ray_init.num_cpus=${RAY_NUM_CPUS})
fi
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

if [ "${RAY_STOP:-1}" = 1 ]; then
    ray stop --force
    sleep 3
fi

python3 -m verl.trainer.main_fastrl \
    speculative.eagle.spec_model_path=$SPEC_MODEL_PATH \
    speculative.enable=${SPEC_ENABLE} \
    speculative.bs_threshold=32 \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/validation.parquet \
    data.return_raw_chat=True \
    data.return_full_prompt=True \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${STEPS} \
    "${EXTRA_ARGS[@]}" \
    "$@"
