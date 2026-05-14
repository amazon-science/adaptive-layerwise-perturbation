#!/bin/bash
# Experiment: GRPO Bypass (bypass old logprob for rollout)
# Usage: bash run_exp_bypass.sh [loss_mode] [dataset] [model]
#
# Prerequisites: conda activate verl_new
#
# Examples:
#   bash run_exp_bypass.sh                                   # Default: sequence + guru + Qwen2.5-Math-1.5B
#   bash run_exp_bypass.sh token guru qwen2.5-1.5b-math     # Token-level + Qwen2.5-Math-1.5B
#   bash run_exp_bypass.sh sequence openr1 qwen3-4b         # Sequence-level + Qwen3-4B
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils_gpu.sh"
source "${SCRIPT_DIR}/env_defaults.sh"

# Parse arguments
LOSS_MODE="${1:-sequence}"               # token/sequence/cum-token/cum-turn
DATASET="${2:-guru}"                     # guru or openr1
MODEL_CHOICE="${3:-qwen2.5-1.5b-math}"  # qwen2.5-1.5b-math or qwen3-4b

# Validate loss mode
if [[ "$LOSS_MODE" != "token" && "$LOSS_MODE" != "sequence" && "$LOSS_MODE" != "cum-token" && "$LOSS_MODE" != "cum-turn" ]]; then
    echo "Error: Invalid loss mode. Use 'token', 'sequence', 'cum-token', or 'cum-turn'"
    exit 1
fi

case "${MODEL_CHOICE}" in
    qwen2.5-1.5b-math|qwen2.5-math-1.5b|qwen2.5)
        MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"
        ;;
    qwen3-4b|qwen3)
        MODEL_PATH="Qwen/Qwen3-4B"
        ;;
    *)
        echo "Error: Invalid model '${MODEL_CHOICE}'. Use 'qwen2.5-1.5b-math' or 'qwen3-4b'."
        exit 1
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"

FREE_GPUS="0,1,2,3,4,5,6,7"
FREE_GPU_COUNT=8
export CUDA_VISIBLE_DEVICES=${FREE_GPUS}
echo "Using fixed GPUs: ${CUDA_VISIBLE_DEVICES}"

# Set temp directories (use repository cache path instead of /tmp to reduce OOM risk).
# Detect if running in Docker container and set paths accordingly.
if [ -d "/workspace/project" ]; then
    # Running in Docker container - use workspace path
    BASE_TMP_DIR="/workspace/project/.cache"
    export RAY_TMPDIR=${BASE_TMP_DIR}/ray_tmp_${FREE_GPU_COUNT}gpu
    ROLLOUT_DUMP_DIR="${BASE_TMP_DIR}/rollout_dump"
else
    # Running locally
    BASE_TMP_DIR="${CACHE_ROOT}"
    export RAY_TMPDIR=${BASE_TMP_DIR}/ray_tmp_${FREE_GPU_COUNT}gpu
    ROLLOUT_DUMP_DIR="${BASE_TMP_DIR}/rollout_dump"
fi

# Set system-wide temp directories to avoid using /tmp
export TMPDIR=${BASE_TMP_DIR}/tmp
export TMP=${TMPDIR}
export TEMP=${TMPDIR}

export NCCL_P2P_DISABLE=1

# Set Ray memory management to prevent OOM
# Increase memory usage threshold from default 0.95 to 0.98
export RAY_memory_usage_threshold=0.98
# Enable automatic cleanup of unused objects
export RAY_enable_auto_cleanup=true
# Set memory monitor refresh interval (ms) - lower value means more frequent checks
export RAY_memory_monitor_refresh_ms=1000
# Limit object store memory to 30GB to prevent excessive memory usage
export RAY_object_store_memory=32212254720
# Enable object store spilling to disk when memory is full
# Using environment variable format that Ray understands
export RAY_object_spilling_config="{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"${RAY_TMPDIR}/spill\"}}"

# Set WandB credentials
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Create all temp directories
mkdir -p $RAY_TMPDIR
mkdir -p ${RAY_TMPDIR}/spill
chmod -R 777 $RAY_TMPDIR 2>/dev/null || true

mkdir -p $TMPDIR
chmod -R 777 $TMPDIR 2>/dev/null || true

# Create rollout dump directory
mkdir -p $ROLLOUT_DUMP_DIR
chmod -R 777 $ROLLOUT_DUMP_DIR 2>/dev/null || true

clip_ratio_low=0.5
clip_ratio_high=3.0

# Shared parameters
    max_prompt_length=$((2048 * 1))
    max_response_length=$((2048))
    train_prompt_bsz=512
    n_resp_per_prompt=8
    train_prompt_mini_bsz=32
    ppo_micro_batch_size=8
    rollout_log_prob_micro_bsz=8
    ref_log_prob_micro_bsz=8
    gpu_memory_util=0.7
    ray_num_cpus=64
    loss_agg_mode="token-mean"

echo "=========================================="
echo "Experiment: GRPO Bypass"
echo "Dataset: ${DATASET}"
echo "Loss Mode: ${LOSS_MODE}"
echo "Model: ${MODEL_PATH}"
echo "Using ${FREE_GPU_COUNT} GPUs: ${FREE_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "----------------------------------------"
echo "Batch Configuration:"
echo "  train_prompt_bsz: ${train_prompt_bsz}"
echo "  n_resp_per_prompt: ${n_resp_per_prompt}"
echo "  train_prompt_mini_bsz: ${train_prompt_mini_bsz}"
echo "  ppo_micro_batch_size: ${ppo_micro_batch_size}"
echo "  rollout_log_prob_micro_bsz: ${rollout_log_prob_micro_bsz}"
echo "  ref_log_prob_micro_bsz: ${ref_log_prob_micro_bsz}"
echo "  gpu_memory_util: ${gpu_memory_util}"
echo "  ray_num_cpus: ${ray_num_cpus}"
echo "----------------------------------------"
echo "Start time: $(date)"
echo "=========================================="

# Resolve dataset file paths from DATA_ROOT
if [ "$DATASET" = "openr1" ]; then
    DATASET_ROOT="${DATA_ROOT}/openr1"
    train_file="${DATASET_ROOT}/train.parquet"
    val_file="${DATASET_ROOT}/test.parquet"
    DATASET_NAME="openr1"
else
    DATASET_ROOT="${DATA_ROOT}/guru_rl92k"
    train_file="${DATASET_ROOT}/train/math__combined_54.4k.parquet"
    val_file="[${DATASET_ROOT}/online_eval/math__math_500.parquet,${DATASET_ROOT}/online_eval/math__aime_repeated_8x_240.parquet]"
    DATASET_NAME="guru_rl92k_math"
fi

# Model configuration
MODEL_ID=$(echo "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g')

# Generate experiment name
project_name="mismatch_rl_research"
EXP_NAME="exp_grpo_bypass_analysis_${MODEL_ID}_${DATASET_NAME}_${LOSS_MODE}_n${n_resp_per_prompt}_bz${train_prompt_bsz}_mini_bz${train_prompt_mini_bsz}"

# Checkpoint directory (handle both local and Docker environments)
if [ -d "/workspace/project" ]; then
    # Running in Docker container
    CKPTS_DIR="/checkpoints/${project_name}/${EXP_NAME}"
else
    # Running locally
    CKPTS_DIR="${CHECKPOINT_ROOT}/${project_name}/${EXP_NAME}"
fi

# Change to project directory (handle both Docker and local environments)
if [ -d "/workspace/project" ]; then
    # Running in Docker container
    cd /workspace/project
else
    # Running locally
    cd ${PROJECT_ROOT}
fi

# Create logs, outputs, and checkpoint directories
mkdir -p logs outputs
chmod 777 logs outputs 2>/dev/null || true
mkdir -p "${CKPTS_DIR}"
chmod -R 777 "${CKPTS_DIR}" 2>/dev/null || true

${PYTHON_BIN} -m verl.trainer.main_ppo \
    hydra.run.dir=outputs/${EXP_NAME}/${now:%Y-%m-%d}/${now:%H-%M-%S} \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_file} \
    data.val_files=${val_file} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.nccl_timeout=1800 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_bsz} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_util} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.skip_dump_dir=${ROLLOUT_DUMP_DIR} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    +algorithm.rollout_correction.rollout_is=null \
    +algorithm.rollout_correction.bypass_old_logprob_for_rollout=true \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${EXP_NAME} \
    +trainer.wandb_entity=${WANDB_ENTITY} \
    +trainer.wandb_mode=${WANDB_MODE} \
    +trainer.wandb_tags=["grpo","bypass","${LOSS_MODE}"] \
    +trainer.wandb_config.loss_mode=${LOSS_MODE} \
    +trainer.wandb_config.clip_ratio_low=${clip_ratio_low} \
    +trainer.wandb_config.clip_ratio_high=${clip_ratio_high} \
    +trainer.wandb_config.loss_agg_mode=${loss_agg_mode} \
    trainer.n_gpus_per_node=${FREE_GPU_COUNT} \
    trainer.nnodes=1 \
    +trainer.ray_init.num_gpus=${FREE_GPU_COUNT} \
    +trainer.ray_init.num_cpus=${ray_num_cpus} \
    +trainer.ray_init.object_store_memory=32212254720 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=50 \
    2>&1 | tee logs/${EXP_NAME}.log

echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="
