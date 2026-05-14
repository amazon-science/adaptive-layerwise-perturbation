dataset_base_path='/home/chenluy/reinforce_flow/data/gen_data/DAPO-Qwen2.5-1.5b-MATH-fix-gen8-continuous'

for step in 500; do
    echo "global_step_$step"
    python compute_score_minerval.py \
        --dataset_path $dataset_base_path/global_step_$step/merged/weqweasdas/minerva_math/merged_data.jsonl \
        --record_path $dataset_base_path/global_step_$step/merged/weqweasdas/minerva_math/new_record.txt
done