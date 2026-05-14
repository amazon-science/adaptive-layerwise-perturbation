
#!/bin/bash

# Configuration
base_output_dir="/home/chenluy/reinforce_flow/data/gen_data/qwen2.5math-1.5b-gen8-global-meanvar-iter"
mkdir -p $base_output_dir

K=16
world_size=4

# Model and dataset arrays
models=()
base_model_path="reinforce-flow/qwen2.5math-1.5b-gen8-global-meanvar-iter"

# 生成从50到350的所有模型路径（步长为50）
for step in $(seq 40 40 600); do
    models+=("$base_model_path-$step")
done

datasets=("weqweasdas/math500" "weqweasdas/minerva_math" "weqweasdas/olympiadbench" "weqweasdas/aime_hmmt_brumo_cmimc_amc23")

# Create base output directory
mkdir -p $base_output_dir

# Loop through models and datasets
for model_name in "${models[@]}"; do
    echo "Testing model: $model_name"
    
    for dataset in "${datasets[@]}"; do
        echo "Testing dataset: $dataset"
        
        # Create model/dataset specific output directory
        # Extract step number from model name
        step=$(echo "$model_name" | sed 's|.*-iter-||')
        output_dir="$base_output_dir/global_step_$step/$dataset"
        mkdir -p "$output_dir"
        
        echo "Output directory: $output_dir"
        
        # Generate data in parallel
        echo "Starting parallel data generation..."
        # we use gpu 4,5,6,7
        for i in 0 1 2 3; do
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
        
        # Compute scores
        echo "Computing scores..."
        python3 compute_score.py \
            --dataset_path "$output_dir/merged_data.jsonl" \
            --record_path "$output_dir/record.txt"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compute scores for $model_name on $dataset"
            continue
        fi
        
        echo "Completed evaluation for $model_name on $dataset"
        echo "Results saved to: $output_dir/record.txt"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"