#!/bin/bash
set -x

# ============================================================================
# G-PPO Training Script
# Combines PPO (Actor/Critic) with GRPO (multiple rollouts, outcome rewards)
# - Uses value baseline V(s_{t-1}) for variance reduction
# - BCE loss for value model (binary outcome prediction)
# - KL penalty as auxiliary loss (not in reward)
# ============================================================================

# --- 1. USER CONFIGURATION (Data, Model, Logs) ---
export RAY_memory_monitor_refresh_ms=0

# Your Log Directory
LOG_DIR="${LOG_DIR:-./logs}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="gppo_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/train_log_${EXPERIMENT_NAME}.log"
mkdir -p "$LOG_DIR"

# Your Model and Data Paths (override these environment variables as needed)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-}"
TEST_DATA="${TEST_DATA:-}"

# --- 2. CORE PARAMETERS (G-PPO Configuration) ---
# PERFORMANCE SETTING: Set to 'true' if you encounter Out Of Memory (OOM) errors.
offload=${OFFLOAD:-false}

project_name='G-PPO'
adv_estimator=gppo
value_loss_type=bce  # BCE loss for binary outcome prediction
rollout_engine=vllm
rollout_mode=sync
gpu_memory_utilization=0.8

# Training schedule
test_freq=${TEST_FREQ:-5}
save_freq=${SAVE_FREQ:-10}
total_epochs=${TOTAL_EPOCHS:-1}
total_training_steps=${TOTAL_TRAINING_STEPS:-100}
val_before_train=false

# KL configuration: use as auxiliary loss, not in reward
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=${USE_KL_LOSS:-true}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
kl_loss_type=low_var_kl

# PPO clipping
clip_ratio_low=0.2
clip_ratio_high=0.28

# Batch sizes
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE:-8}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}  # Multiple rollouts like GRPO

# Context Lengths
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

# Checkpoint directory
CKPTS_DIR="${CKPTS_DIR:-./checkpoints/${EXPERIMENT_NAME}}"
mkdir -p "$CKPTS_DIR"

# Sampling params
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.95

# Performance Related Parameters
sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
critic_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
gen_tp=1

# Number of GPUs
N_GPUS=${N_GPUS:-1}
N_NODES=${N_NODES:-1}

# --- 3. EXECUTION ---
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.value_loss_type=${value_loss_type} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files="['$TRAIN_DATA']" \
    data.val_files="['$TEST_DATA']" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    critic.optim.lr=2e-6 \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_max_token_len_per_gpu=${critic_ppo_max_token_len} \
    critic.ulysses_sequence_parallel_size=${sp_size} \
    critic.model.fsdp_config.param_offload=${offload} \
    critic.model.fsdp_config.optimizer_offload=${offload} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${N_NODES} \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    $@ 2>&1 | tee "$LOG_FILE"
