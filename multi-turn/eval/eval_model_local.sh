
#!/bin/bash

# Configuration
base_output_dir="${PROJECT_DIR:-.}/eval/gen_data/simpletir_cliph3.0_clipl0.5_clipc5.0_maxturn5_maskvoidTrue_oversample1_losssequence_adaptratioFalse_ppogeoFalse_risTrue_isth2.0_isthlower0.0_isveto0_islvlsequence_batch256_ppomini32_deepscaler_merge_train_Qwen2.5-7B"
mkdir -p $base_output_dir

K=32
world_size=8

# Base model path (for tokenizer and config)
BASE_MODEL_PATH="Qwen/Qwen2.5-7B"

# Model and dataset arrays
models=()
base_model_path="${CHECKPOINT_DIR}/simpletir_cliph3.0_clipl0.5_clipc5.0_maxturn5_maskvoidTrue_oversample1_losssequence_adaptratioFalse_ppogeoFalse_risTrue_isth2.0_isthlower0.0_isveto0_islvlsequence_batch256_ppomini32_deepscaler_merge_train_Qwen2.5-7B"

# merge the model
echo "=== Starting model merging ==="
for step in $(seq 200 20 300); do
    actor_dir="$base_model_path/global_step_$step/actor"
    merged_dir="$base_model_path/global_step_$step/merged"
    
    if [ -d "$merged_dir" ]; then
        echo "✓ Model for step $step already merged"
        continue
    fi
    
    echo "Merging model for step $step..."
    python3 ${PROJECT_DIR:-.}/scripts/model_merger.py \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --hf_model_path "$BASE_MODEL_PATH" \
        --target_dir "$merged_dir"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully merged step $step"
        
        # Copy tokenizer files from base model to merged directory
        echo "  Copying tokenizer files from $BASE_MODEL_PATH to merged directory..."
        python3 -c "
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL_PATH', trust_remote_code=True)
tokenizer.save_pretrained('$merged_dir')
print('✓ Tokenizer files copied to merged directory')
"
        if [ $? -ne 0 ]; then
            echo "✗ Failed to copy tokenizer files"
            exit 1
        fi
    else
        echo "✗ Failed to merge step $step"
        exit 1
    fi
done

# Generate model paths for global_step_20 to global_step_220 (increment by 20)
for step in $(seq 200 20 300); do
    models+=("$base_model_path/global_step_$step/merged")
done

datasets=("deepscaler/math500" "deepscaler/minerva_math" "deepscaler/olympiadbench" "deepscaler/aime" "deepscaler/aime25" "deepscaler/hmmt0225") 

# Create base output directory
mkdir -p $base_output_dir

# Loop through models and test on all datasets at once
for model_name in "${models[@]}"; do
    echo "Testing model: $model_name"
    echo "Testing datasets: ${datasets[*]}"
    
    # Create model specific output directory (without /merged suffix)
    # Extract global_step_X from the full path
    model_step=$(echo "$model_name" | grep -oP 'global_step_[0-9]+')
    output_dir="$base_output_dir/$model_step"
    mkdir -p "$output_dir"
    
    echo "Output directory: $output_dir"
    
    # Extract model parent directory and model name
    model_parent_dir=$(dirname "$model_name")
    model_basename=$(basename "$model_name")
    
    # Run validation using train.sh with all datasets
    echo "Starting validation using train.sh..."
    cd ${PROJECT_DIR:-.}
    
    # Disable wandb for validation
    export WANDB_MODE=disabled
    
    MODEL_PATH="$model_parent_dir" \
    DATA_PATH=./datasets \
    CHECKPOINT_PATH="$output_dir" \
    NNODES=1 \
    GPUS_PER_NODE=8 \
    RESUME=False \
    CONFIG_NAME=simpletir_trainer \
    bash train.sh \
      --max_response_length 8000 \
      --max_prompt_length 16000 \
      --model_path "$model_parent_dir" \
      --model_name "$model_basename" \
      --max_turns 5 \
      --valid_dataset "${datasets[*]}" \
      --val_only True \
      --n_val $K \
      --output_acc_to_file True \
      --val_sample_size null \
      --val_batch_size 256 \
      --sp_size 1 \
      --total_epochs 1 \
      --val_temperature 1.0
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run validation for $model_name"
        continue
    fi
    
    echo "Completed evaluation for $model_name on all datasets"
    echo "Results saved to: $output_dir"
    echo "----------------------------------------"
done

echo "All evaluations completed!"