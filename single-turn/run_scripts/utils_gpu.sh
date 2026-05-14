#!/bin/bash
# GPU utility functions

# Get comma-separated list of free GPU IDs
# A GPU is free if: util < 10% AND mem_used < 1000 MB
get_free_gpus() {
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F, '{gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $3); if ($2 < 10 && $3 < 1000) print $1}' | \
    paste -sd,
}

# Get count of free GPUs
get_free_gpu_count() {
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F, '{gsub(/^[ \t]+|[ \t]+$/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $3); if ($2 < 10 && $3 < 1000) print}' | \
    wc -l
}

# Check if enough free GPUs available
check_min_gpus() {
    local required=$1
    local available=$(get_free_gpu_count)
    
    if [ $available -lt $required ]; then
        echo "ERROR: Need $required GPUs, but only $available are free"
        echo "Currently free GPUs: $(get_free_gpus)"
        echo ""
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
        return 1
    fi
    return 0
}

# Export this for sourcing
export -f get_free_gpus
export -f get_free_gpu_count
export -f check_min_gpus

