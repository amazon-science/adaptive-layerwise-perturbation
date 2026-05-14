#!/bin/bash
set -e

# Set temp directories to NVMe disk to avoid OOM
export TMPDIR=${CACHE_DIR:-/tmp}
export TEMP=${CACHE_DIR:-/tmp}
export TMP=${CACHE_DIR:-/tmp}

# PyTorch shared memory path (used for multiprocess communication)
export TORCH_SHARED_MEMORY_DIR=${CACHE_DIR:-/tmp}/torch_shm

# Set HuggingFace cache directories (all using the same path)
export HF_HOME=${CACHE_DIR:-/tmp}/hf_cache
export HF_HUB_CACHE=${CACHE_DIR:-/tmp}/hf_cache
export TRANSFORMERS_CACHE=${CACHE_DIR:-/tmp}/hf_cache
export HF_DATASETS_CACHE=${CACHE_DIR:-/tmp}/hf_cache
export HF_DATASETS_OFFLINE=0
# Ensure using local cache
export HF_HUB_OFFLINE=0

# Set PyTorch temp directory
export TORCH_HOME=${CACHE_DIR:-/tmp}/torch_cache

# Create necessary directories
mkdir -p $TMPDIR
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p ${CACHE_DIR:-/tmp}/axolotl_prepared_cache
mkdir -p ${CACHE_DIR:-/tmp}/torch_shm

# Clean up old PyTorch shared memory files (if present)
echo "Cleaning old temp files..."
find ${CACHE_DIR:-/tmp}/torch_shm -name "torch_*" -type f -mtime +1 -delete 2>/dev/null || true

# Clean torch-related files in /dev/shm and /tmp
echo "Cleaning torch files in /dev/shm..."
find /dev/shm -name "torch_*" -type f -delete 2>/dev/null || true
find /dev/shm -name "sem.torch*" -delete 2>/dev/null || true

echo "Cleaning torch files in /tmp..."
find /tmp -name "torch_*" -type f -delete 2>/dev/null || true
find /tmp -name "tmp*" -type f -mtime +1 -delete 2>/dev/null || true

# Increase file descriptor limit
ulimit -n 65536 2>/dev/null || true

# Check disk space
echo "Checking disk space:"
df -h ${CACHE_DIR:-/tmp} | tail -1

# Pre-download model (avoid conflicts from concurrent downloads)
# Use huggingface-cli for lightweight download without loading into memory
echo "Checking/pre-downloading model Qwen/Qwen2.5-7B..."
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download Qwen/Qwen2.5-7B --cache-dir $HF_HOME --local-dir-use-symlinks False || echo "Model may already exist or download failed, continuing..."
else
    echo "huggingface-cli not installed, skipping pre-download (will auto-download during training)"
fi

# Set other potentially needed environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Disable PyTorch shared memory (use filesystem to avoid IPC issues)
export TORCH_SHARED_MEMORY_ALLOCATION=file_system

# Set DataLoader environment variables (if supported by axolotl)
export DATALOADER_NUM_WORKERS=0
export DATALOADER_PIN_MEMORY=false

# Limit PyTorch multiprocess shared memory usage
export PYTORCH_MULTIPROCESSING_SHARING_STRATEGY=file_descriptor

echo "Temp directory settings:"
echo "  TMPDIR=$TMPDIR"
echo "  HF_HOME=$HF_HOME"
echo "  TORCH_HOME=$TORCH_HOME"
echo ""

# Run axolotl training
accelerate launch -m axolotl.cli.train qwen2-5_7b.yaml

