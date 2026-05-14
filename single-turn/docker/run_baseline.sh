#!/bin/bash
# Docker wrapper for baseline experiment
# Usage: bash docker/run_baseline.sh [dataset]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
DATASET="${1:-guru}"  # guru or openr1

# Get GPU configuration from environment or use default
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    HOST_GPUS="$CUDA_VISIBLE_DEVICES"
    GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    HOST_GPUS="0,1,2,3"
    GPU_COUNT=4
fi

# Container will see GPUs as 0,1,2,... so we set CUDA_VISIBLE_DEVICES accordingly
CONTAINER_CUDA_VISIBLE=$(seq -s, 0 $((GPU_COUNT-1)))

# Match container user to host user to avoid permission issues
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(id -un)

echo "=========================================="
echo "Baseline 实验 Docker 启动器"
echo "Dataset: ${DATASET}"
echo "宿主机 GPU: ${HOST_GPUS}"
echo "容器内将看到 ${GPU_COUNT} 个GPU (编号 ${CONTAINER_CUDA_VISIBLE})"
echo "=========================================="

# Fix permissions
bash "$SCRIPT_DIR/fix_permissions.sh"
echo ""

# Build image if needed
if ! docker image inspect verl-mismatch:latest &> /dev/null; then
    echo "构建 Docker 镜像..."
    docker build -t verl-mismatch:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_DIR"
fi

# Run container
# Use specific GPUs and set CUDA_VISIBLE_DEVICES correctly inside container
GPU_DEVICE_ARG="\"device=${HOST_GPUS}\""

HOST_DATA_ROOT=${HOST_DATA_ROOT:-/home/zhang430/data/guru_rl92k}
CONTAINER_DATA_ROOT=/data/guru_rl92k

HOST_TRAIN_FILE="${HOST_DATA_ROOT}/train/math__combined_54.4k.parquet"
HOST_VAL_FILES=(
    "${HOST_DATA_ROOT}/online_eval/math__math_500.parquet"
    "${HOST_DATA_ROOT}/online_eval/math__aime_repeated_8x_240.parquet"
)

CONTAINER_TRAIN_FILE="${CONTAINER_DATA_ROOT}/train/math__combined_54.4k.parquet"
CONTAINER_VAL_FILES=(
    "${CONTAINER_DATA_ROOT}/online_eval/math__math_500.parquet"
    "${CONTAINER_DATA_ROOT}/online_eval/math__aime_repeated_8x_240.parquet"
)

export HOST_DATA_ROOT CONTAINER_DATA_ROOT

if [ ! -f "${HOST_TRAIN_FILE}" ]; then
    echo "Error: train file ${HOST_TRAIN_FILE} not found" >&2
    exit 1
fi
for HOST_VAL_FILE in "${HOST_VAL_FILES[@]}"; do
    if [ ! -f "${HOST_VAL_FILE}" ]; then
        echo "Error: val file ${HOST_VAL_FILE} not found" >&2
        exit 1
    fi
done

if [ -z "${TRAIN_FILES_JSON}" ]; then
    TRAIN_FILES_JSON="[\"${CONTAINER_TRAIN_FILE}\"]"
fi

if [ -z "${VAL_FILES_JSON}" ]; then
    VAL_FILES_JSON="[\"${CONTAINER_VAL_FILES[0]}\",\"${CONTAINER_VAL_FILES[1]}\"]"
fi

export TRAIN_FILES_JSON VAL_FILES_JSON

docker run --rm \
    --name verl_baseline_exp \
    --gpus "${GPU_DEVICE_ARG}" \
    --privileged \
    --user "${HOST_UID}:${HOST_GID}" \
    --network host \
    --ipc host \
    --shm-size=32gb \
    -e USER=${HOST_USER} \
    -e LOGNAME=${HOST_USER} \
    -e HOME=/workspace/project \
    -e CUDA_VISIBLE_DEVICES=${CONTAINER_CUDA_VISIBLE} \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e WANDB_API_KEY=YOUR_WANDB_KEY \
    -e WANDB_ENTITY=mismatch \
    -e PYTHONPATH=/workspace/verl:/workspace/project \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e VLLM_USE_V1=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e TRAIN_FILES_JSON="${TRAIN_FILES_JSON}" \
    -e VAL_FILES_JSON="${VAL_FILES_JSON}" \
    -v "$PROJECT_DIR:/workspace/project" \
    -v "/home/zhang430/data:/data:ro" \
    -v "/home/zhang430/checkpoints:/checkpoints" \
    -v verl_huggingface_cache:/workspace/.cache/huggingface \
    -w /workspace/project \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536 \
    verl-mismatch:latest \
    bash -c "
        set -e
        echo '容器内环境检查:'
        python -c 'import torch; print(\"  PyTorch:\", torch.__version__)'
        python -c 'import vllm; print(\"  vLLM:\", vllm.__version__)'
        python -c 'import torch; print(\"  CUDA:\", torch.cuda.is_available(), \"GPUs:\", torch.cuda.device_count())'
        echo ''
        
        ray stop --force 2>/dev/null || true
        sleep 2
        
        mkdir -p /workspace/project/outputs /workspace/project/logs /checkpoints/mismatch_rl_research
        
        echo '开始运行实验...'
        cd /workspace/project/run_scripts
        bash run_exp_baseline.sh ${DATASET}
    "

