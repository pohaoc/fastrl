#!/bin/bash
# In-loop smoke test on 4x L40S: upstream examples/grpo_7B.sh, 1 step, SD on, attention backend $ATTN.
# The draft CUDA graphs of every MAB strategy are captured when the rollout engine starts, which is where
# fa3 crashed in-loop before (jobs 6747961/6749022), so one step answers whether fa3 works here.
#   sbatch reproducibility/oscar/grpo_smoke.sbatch                    (submit from the repo root)
#   ATTN=triton TP=2 sbatch reproducibility/oscar/grpo_smoke.sbatch
# Deviations from examples/grpo_7B.sh are marked "# DEV:".
#SBATCH --job-name=grpo-smoke-l40s
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu3001                 # uncorrectable ECC errors (2026-09-30, 2026-10-04)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -euo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source reproducibility/oscar/env.sh
ATTN=${ATTN:-fa3}
TP=${TP:-4}
SD=${SD:-true}
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) commit=$(git rev-parse --short HEAD) ATTN=$ATTN TP=$TP SD=$SD"
git status --short
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-Coder-7B-Instruct', revision='c03e6d358207e414f1eca0bb1891e29f1db0e242'))")
SPEC_MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download as s; print(s('mit-han-lab/Qwen2.5-7B-Eagle-RL', revision='38905764986fdd43c64f0b5ab2a0bdac6bf71b08'))")
export HF_HUB_OFFLINE=1

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export MKL_SERVICE_FORCE_INTEL=1
# Ray: own temp dir (Slurm wipes /tmp at job end, so failures copy the logs out) and never `ray stop` others.
export RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID
mkdir -p "$RAY_TMPDIR"
OUT=$RUN_ROOT/grpo_smoke/$SLURM_JOB_ID
mkdir -p "$OUT"

PROJECT_NAME=FastRL
EXPERIMENT_NAME=smoke-coder7b-sd${SD}-${ATTN}-tp${TP}
DATA_PATH=$DATA_ROOT/Eurus-2-RL-Data       # upstream's dataset; rollout content is irrelevant to this test

train_prompt_bsz=8                         # DEV: 64 -> 8 (smoke); 8x8 = 64 requests still drains through all MAB bs ranges
n_resp_per_prompt=8
train_prompt_mini_bsz=4
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 4))          # DEV: 32K -> 4K (smoke; Coder-7B's max_position is 32768 anyway)
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

set +e
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
    actor_rollout_ref.actor.use_kl_loss=True \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=$ATTN \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUT/ckpt \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    ray_init.num_cpus=$SLURM_CPUS_PER_TASK
rc=$?
# DEV (lines above): engine_kwargs attention_backend (verl forces fa3 when unset, so ATTN=fa3 == upstream),
# n_gpus_per_node 8 -> 4, save_freq 30 -> -1, total_training_steps=1, ray_init.num_cpus (Oscar cgroup).
if [ $rc -ne 0 ]; then
    cp -r "$RAY_TMPDIR"/ray/session_latest/logs "$OUT/ray_logs" 2>/dev/null || cp -r "$RAY_TMPDIR"/session_latest/logs "$OUT/ray_logs" 2>/dev/null
fi
echo "exit=$rc end=$(date)"
exit $rc
