#!/bin/bash
# Unified Experiment: GRPO + Truncated Importance Sampling (TIS)
# Usage: bash run_exp_tis.sh [level] [mode] [threshold] [veto_threshold] [dataset] [model]
#
# Prerequisites: conda activate verl_new
#
# Examples:
#   bash run_exp_tis.sh token truncate 5.0 null guru qwen2.5-1.5b-math
#   bash run_exp_tis.sh sequence mask 3.0 0.001 openr1 qwen3-4b
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils_gpu.sh"
source "${SCRIPT_DIR}/env_defaults.sh"

# Parse arguments
TIS_LEVEL="${1:-sequence}"               # token/sequence
TIS_MODE="${2:-truncate}"                # truncate/mask/geometric
TIS_THRESHOLD="${3:-5.0}"                # Upper threshold
VETO_THRESHOLD="${4:-null}"              # Veto threshold (null=disabled)
DATASET="${5:-guru}"                     # guru or openr1
MODEL_CHOICE="${6:-qwen2.5-1.5b-math}"  # qwen2.5-1.5b-math or qwen3-4b

# Validate level
if [[ "$TIS_LEVEL" != "token" && "$TIS_LEVEL" != "sequence" ]]; then
    echo "Error: Invalid TIS level. Use 'token' or 'sequence'"
    exit 1
fi

# Validate mode
if [[ "$TIS_MODE" != "truncate" && "$TIS_MODE" != "mask" && "$TIS_MODE" != "geometric" ]]; then
    echo "Error: Invalid TIS mode. Use 'truncate', 'mask', or 'geometric'"
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

# Get GPU configuration (fixed to 8 GPUs)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    FREE_GPUS=$CUDA_VISIBLE_DEVICES
else
    FREE_GPUS="0,1,2,3,4,5,6,7"
fi
FREE_GPU_COUNT=$(echo $FREE_GPUS | tr ',' '\n' | wc -l)
if [ "$FREE_GPU_COUNT" -ne 8 ]; then
    echo "Warning: Expected 8 GPUs, but got ${FREE_GPU_COUNT} from CUDA_VISIBLE_DEVICES=${FREE_GPUS}"
fi

# Only export CUDA_VISIBLE_DEVICES when running outside Docker
if [ ! -d "/workspace/project" ]; then
    export CUDA_VISIBLE_DEVICES=${FREE_GPUS}
fi

# Configure rollout correction parameters based on mode
if [ "$TIS_MODE" = "truncate" ]; then
    # Truncate mode: Use IS weights only (no rejection sampling)
    ROLLOUT_IS="${TIS_LEVEL}"
    ROLLOUT_RS="null"
    RS_THRESHOLD="null"
elif [ "$TIS_MODE" = "mask" ]; then
    # Mask mode: Use IS weights + rejection sampling at same level
    ROLLOUT_IS="${TIS_LEVEL}"
    ROLLOUT_RS="${TIS_LEVEL}"
    RS_THRESHOLD="${TIS_THRESHOLD}"
elif [ "$TIS_MODE" = "geometric" ]; then
    # Geometric mode: Use geometric rejection sampling (no IS)
    ROLLOUT_IS="null"
    ROLLOUT_RS="geometric"
    RS_THRESHOLD="${TIS_THRESHOLD}"
fi

export RAY_TMPDIR=${CACHE_ROOT}/ray_tmp_${FREE_GPU_COUNT}gpu
export NCCL_P2P_DISABLE=1

mkdir -p $RAY_TMPDIR
chmod -R 777 $RAY_TMPDIR 2>/dev/null || true

clip_ratio_low=0.2
clip_ratio_high=0.28

# Shared parameters
max_prompt_length=$((2048 * 1))
max_response_length=$((2048))
train_prompt_bsz=512
n_resp_per_prompt=8
train_prompt_mini_bsz=32
ppo_micro_batch_size=8
rollout_log_prob_micro_bsz=8
ref_log_prob_micro_bsz=8
gpu_memory_util=0.8
ray_num_cpus=64
loss_agg_mode="token-mean"

echo "=========================================="
echo "Experiment: GRPO + TIS"
echo "Dataset: ${DATASET}"
echo "IS Level: ${ROLLOUT_IS}"
echo "RS Mode: ${ROLLOUT_RS}"
echo "IS Threshold: ${TIS_THRESHOLD}"
echo "RS Threshold: ${RS_THRESHOLD}"
echo "Veto: ${VETO_THRESHOLD}"
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
    val_files_json='["'"${val_file}"'"]'
    DATASET_NAME="openr1"
else
    DATASET_ROOT="${DATA_ROOT}/guru_rl92k"
    train_file="${DATASET_ROOT}/train/math__combined_54.4k.parquet"
    val_file_math="${DATASET_ROOT}/online_eval/math__math_500.parquet"
    val_file_aime="${DATASET_ROOT}/online_eval/math__aime_repeated_8x_240.parquet"
    val_files_json='["'"${val_file_math}"'","'"${val_file_aime}"'"]'
    DATASET_NAME="guru_rl92k_math"
fi

if [ ! -f "${train_file}" ]; then
    echo "Error: train file ${train_file} not found" >&2
    exit 1
fi

if [ "$DATASET" = "openr1" ]; then
    if [ ! -f "${val_file}" ]; then
        echo "Error: val file ${val_file} not found" >&2
        exit 1
    fi
else
    for vf in "${val_file_math}" "${val_file_aime}"; do
        if [ ! -f "${vf}" ]; then
            echo "Error: val file ${vf} not found" >&2
            exit 1
        fi
    done
fi

# Model & dataset configuration
MODEL_ID=$(echo "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g')

# Generate experiment name
project_name="mismatch_rl_research"
EXP_NAME="exp_tis_${MODEL_ID}_${DATASET_NAME}_${TIS_LEVEL}_${TIS_MODE}_th${TIS_THRESHOLD}_n${n_resp_per_prompt}_bz${train_prompt_bsz}_mini_bz${train_prompt_mini_bsz}"
if [ "${ROLLOUT_RS}" != "null" ] && [ "${RS_THRESHOLD}" != "null" ]; then
    EXP_NAME="${EXP_NAME}_rs${RS_THRESHOLD}"
fi
if [ "$VETO_THRESHOLD" != "null" ]; then
    EXP_NAME="${EXP_NAME}_veto${VETO_THRESHOLD}"
fi

# Checkpoint directory (handle both local and Docker environments)
if [ -d "/workspace/project" ]; then
    # Running in Docker container
    CKPTS_DIR="/checkpoints/${project_name}/${EXP_NAME}"
else
    # Running locally
    CKPTS_DIR="${CHECKPOINT_ROOT}/${project_name}/${EXP_NAME}"
fi

# Change to project directory (handle both local and Docker environments)
if [ -d "/workspace/project" ]; then
    cd /workspace/project
else
    cd ${PROJECT_ROOT}
fi

# Create logs directory if it doesn't exist
mkdir -p logs

${PYTHON_BIN} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_file} \
    data.val_files=${val_files_json} \
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
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_bsz} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_util} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    +algorithm.rollout_correction.rollout_is=${ROLLOUT_IS} \
    +algorithm.rollout_correction.rollout_is_threshold=${TIS_THRESHOLD} \
    +algorithm.rollout_correction.rollout_rs=${ROLLOUT_RS} \
    +algorithm.rollout_correction.rollout_rs_threshold=${RS_THRESHOLD} \
    +algorithm.rollout_correction.rollout_token_veto_threshold=${VETO_THRESHOLD} \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${EXP_NAME} \
    +trainer.wandb_entity=${WANDB_ENTITY} \
    +trainer.wandb_mode=${WANDB_MODE} \
    +trainer.wandb_tags=["grpo","tis","${TIS_LEVEL}","${TIS_MODE}"] \
    +trainer.wandb_config.tis_level=${TIS_LEVEL} \
    +trainer.wandb_config.tis_mode=${TIS_MODE} \
    +trainer.wandb_config.tis_threshold=${TIS_THRESHOLD} \
    +trainer.wandb_config.veto_threshold=${VETO_THRESHOLD} \
    +trainer.wandb_config.rollout_is=${ROLLOUT_IS} \
    +trainer.wandb_config.rollout_rs=${ROLLOUT_RS} \
    +trainer.wandb_config.rs_threshold=${RS_THRESHOLD} \
    +trainer.wandb_config.clip_ratio_low=${clip_ratio_low} \
    +trainer.wandb_config.clip_ratio_high=${clip_ratio_high} \
    +trainer.wandb_config.loss_agg_mode=${loss_agg_mode} \
    trainer.n_gpus_per_node=${FREE_GPU_COUNT} \
    trainer.nnodes=1 \
    +trainer.ray_init.num_gpus=${FREE_GPU_COUNT} \
    +trainer.ray_init.num_cpus=${ray_num_cpus} \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=50 \
    2>&1 | tee logs/${EXP_NAME}.log

echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="
