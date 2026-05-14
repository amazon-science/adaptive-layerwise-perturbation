#!/bin/bash
# Script to run experiments in Docker with Flash Attention support
# Usage: bash docker/run_docker.sh [--rebuild] [script_name] [args...]
#
# Examples:
#   bash docker/run_docker.sh --rebuild                               # Build/rebuild image
#   bash docker/run_docker.sh run_exp_baseline.sh false               # Run baseline with Flash Attn
#   bash docker/run_docker.sh run_exp_tis.sh sequence truncate 5.0    # Run TIS experiment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "VERL Mismatch Docker 实验运行器"
echo "基于 PyTorch 2.8.0 + CUDA 12.9 + vLLM 0.11.0 + Flash Attention"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装"
    exit 1
fi

# Detect docker-compose command (support both v1 and v2)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "错误: docker-compose 未安装"
    exit 1
fi

# Check NVIDIA Docker runtime
echo "检查 NVIDIA Docker 运行时..."
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
    echo "警告: NVIDIA Docker 运行时可能未正确配置"
    echo "如果遇到 GPU 问题，请安装 nvidia-container-toolkit 并重启 Docker"
    echo "继续执行..."
fi

cd "$PROJECT_DIR"

# Build the Docker image if it doesn't exist or if --rebuild is specified
if [ "$1" == "--rebuild" ] || ! docker image inspect verl-mismatch:latest &> /dev/null; then
    echo "构建 Docker 镜像（这可能需要 15-30 分钟）..."
    echo "基础镜像: hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0"
    $DOCKER_COMPOSE -f docker/docker-compose.yml build --no-cache
    echo "✓ Docker 镜像构建成功"
    
    # If --rebuild was the only argument, exit after building
    if [ "$1" == "--rebuild" ] && [ $# -eq 1 ]; then
        exit 0
    fi
    
    # Shift arguments if --rebuild was first
    if [ "$1" == "--rebuild" ]; then
        shift
    fi
fi

# Parse experiment arguments
SCRIPT_NAME="${1:-run_exp_baseline.sh}"
shift || true  # Remove script name from args
SCRIPT_ARGS="$@"

echo ""
echo "实验配置:"
echo "  - 脚本: ${SCRIPT_NAME}"
echo "  - 参数: ${SCRIPT_ARGS}"
echo ""

# Check GPU availability
echo "检查 GPU 可用性..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
else
    echo "警告: nvidia-smi 不可用，跳过 GPU 检查"
fi
echo ""

# Determine GPU devices (use 0-7 by default)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Ensure docker-compose runs container with host user permissions
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
export LOCAL_USER=$(id -un)
export LOCAL_LOGNAME=${LOCAL_USER}

echo "启动 Docker 容器并运行实验..."
echo "使用 GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

$DOCKER_COMPOSE -f docker/docker-compose.yml run --rm verl-exp bash -c "
    set -e
    
    echo '容器内环境检查:'
    python -c 'import torch; print(\"  - PyTorch:\", torch.__version__)'
    python -c 'import vllm; print(\"  - vLLM:\", vllm.__version__)'
    python -c 'import torch; print(\"  - CUDA 可用:\", torch.cuda.is_available())'
    python -c 'import torch; print(\"  - GPU 数量:\", torch.cuda.device_count())'
    python -c 'import flash_attn; print(\"  - Flash Attention:\", flash_attn.__version__)' || echo '  - Flash Attention: Not installed'
    echo ''
    
    # Stop any existing Ray instance
    echo '清理现有的 Ray 实例...'
    ray stop --force 2>/dev/null || true
    sleep 2
    
    # Run the experiment
    echo '开始运行实验...'
    cd run_scripts
    bash ${SCRIPT_NAME} ${SCRIPT_ARGS}
"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 实验成功完成"
    echo ""
    echo "查看结果:"
    echo "  - 日志: /home/zhang430/code/mismatch_rl/logs/"
    echo "  - WandB: https://wandb.ai/mismatch/mismatch_rl_research"
else
    echo "✗ 实验失败，退出码: $EXIT_CODE"
    echo ""
    echo "故障排查:"
    echo "  1. 检查日志文件获取详细错误信息"
    echo "  2. 确保有足够的 GPU 内存（至少 4 个 GPU）"
    echo "  3. 验证数据文件路径: /home/zhang430/data/openr1/"
fi
echo "=========================================="

exit $EXIT_CODE

