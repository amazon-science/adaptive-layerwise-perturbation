base=/home/wx13/reinforceflow/verl/checkpoints/DAPO/GRPO-baseline-n8-bz512-256-mathbase/global_step_

for i in {20..220..20}; do
cp $base$i/actor/huggingface/* $base$i/actor/
python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir $base$i/actor \
    --target_dir $base$i/merged 
done        