
#!/bin/bash

# Configuration
base_output_dir="/home/chenluy/reinforce_flow/data/gen_data/Reinforce-Ada_square-root-p_Llama-3.2-3B-Instruct_hard_prompt_min4max32_noglobal_adv_sqrt_p_chenlu"
mkdir -p $base_output_dir

K=16
world_size=8

# Model and dataset arrays
models=()
base_model_path="/opt/dlami/nvme/chenluy_ckpoints/Reinforce-Ada/Reinforce-Ada_square-root-p_Llama-3.2-3B-Instruct_hard_prompt_min4max32_noglobal_adv_sqrt_p_chenlu"

# merge the model
for step in $(seq 300 50 500); do
    python /home/chenluy/reinforce_flow/scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir $base_model_path/global_step_$step/actor \
        --hf_model_path $base_model_path/global_step_$step/actor/huggingface \
        --target_dir $base_model_path/global_step_$step/merged
done

# Generate model paths for global_step_20 to global_step_220 (increment by 20)
for step in $(seq 300 50 500); do
    models+=("$base_model_path/global_step_$step/merged")
done

# Only test minerva_math dataset
dataset="weqweasdas/minerva_math"

# Create base output directory
mkdir -p $base_output_dir

# Loop through models
for model_name in "${models[@]}"; do
    echo "Testing model: $model_name"
    echo "Testing dataset: $dataset"
    
    # Create model/dataset specific output directory
    # Extract global_step_X/merged from the full path
    model_step_dir=$(echo "$model_name" | sed 's|.*/\(global_step_[0-9]*/merged\)|\1|')
    output_dir="$base_output_dir/$model_step_dir/$dataset"
    mkdir -p "$output_dir"
    
    echo "Output directory: $output_dir"
    
    # Generate data in parallel
    echo "Starting parallel data generation..."
    # we use gpu 0-7
    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i python3 gen_data.py \
            --local_index $((i)) \
            --my_world_size $world_size \
            --model_name_or_path "$model_name" \
            --output_dir "$output_dir/" \
            --K $K \
            --dataset_name_or_path "$dataset" &
    done
    
    # Wait for all parallel processes to complete
    wait
    echo "Data generation completed."
    
    # Merge the generated data
    echo "Merging data..."
    python3 merge_data.py \
        --base_path "$output_dir/" \
        --output_dir "$output_dir/merged_data.jsonl" \
        --num_datasets $world_size
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to merge data for $model_name on $dataset"
        continue
    fi
    
    # Compute minerva_math scores using compute_score_minerval.py
    echo "Computing minerva_math scores..."
    python3 compute_score_minerval.py \
        --dataset_path "$output_dir/merged_data.jsonl" \
        --record_path "$output_dir/record_new.txt"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to compute scores for $model_name on $dataset"
        continue
    fi
    
    echo "Completed evaluation for $model_name on $dataset"
    echo "Results saved to: $output_dir/record_new.txt"
    echo "----------------------------------------"
done

echo "All evaluations completed!"