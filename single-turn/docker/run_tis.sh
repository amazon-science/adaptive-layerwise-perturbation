#!/bin/bash
# --------------------------------------------------------
# Docker launcher for TIS experiment (8-GPU version)
# ✅ Hosts GPUs 0–7 mapped to container GPUs 0–7
# --------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ----------------------------
# Parse arguments
# ----------------------------
TIS_LEVEL="${1:-sequence}"
TIS_MODE="${2:-truncate}"
TIS_THRESHOLD="${3:-5.0}"
VETO_THRESHOLD="${4:-null}"

# ----------------------------
# GPU selection
# ----------------------------
HOST_GPUS="0,1,2,3,4,5,6,7"
GPU_COUNT=8
CONTAINER_CUDA_VISIBLE=$(seq -s, 0 $((GPU_COUNT-1)))

# ----------------------------
# User configuration
# ----------------------------
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(id -un)

echo "=========================================="
echo "TIS 实验 Docker 启动器"
echo "Level: ${TIS_LEVEL}, Mode: ${TIS_MODE}"
echo "Threshold: ${TIS_THRESHOLD}, Veto: ${VETO_THRESHOLD}"
echo "宿主机 GPU: ${HOST_GPUS}"
echo "容器内映射为: ${CONTAINER_CUDA_VISIBLE} (${GPU_COUNT}个GPU)"
echo "=========================================="

# ----------------------------
# Fix permissions
# ----------------------------
bash "$SCRIPT_DIR/fix_permissions.sh"
echo ""

# ----------------------------
# Build image if needed
# ----------------------------
if ! docker image inspect verl-mismatch:latest &> /dev/null; then
    echo "构建 Docker 镜像..."
    docker build -t verl-mismatch:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_DIR"
fi

# ----------------------------
# Run container
# ----------------------------
# Use --gpus device to select host GPUs 0-7, they will be remapped as 0-7 in container
echo "配置容器使用宿主机GPU: ${HOST_GPUS} → 容器内GPU: ${CONTAINER_CUDA_VISIBLE}"
GPU_DEVICE_ARG='"device='${HOST_GPUS}'"'
docker run --rm \
    --name verl_tis_exp \
    --gpus ${GPU_DEVICE_ARG} \
    --privileged \
    --user "${HOST_UID}:${HOST_GID}" \
    --network host \
    --ipc host \
    --shm-size=32gb \
    -e USER=${HOST_USER} \
    -e LOGNAME=${HOST_USER} \
    -e HOME=/workspace/project \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e WANDB_API_KEY=YOUR_WANDB_KEY \
    -e WANDB_ENTITY=mismatch \
    -e PYTHONPATH=/workspace/verl:/workspace/project \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e VLLM_USE_V1=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e HF_HOME=/workspace/.cache/huggingface \
    -v "$PROJECT_DIR:/workspace/project" \
    -v "/home/zhang430/data:/data:ro" \
    -v "/home/zhang430/checkpoints:/checkpoints" \
    -v "/home/zhang430/code/mismatch_rl/.cache/huggingface:/workspace/.cache/huggingface" \
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
        bash run_exp_tis.sh ${TIS_LEVEL} ${TIS_MODE} ${TIS_THRESHOLD} ${VETO_THRESHOLD}
    "