#!/bin/bash
set -e  # Exit on any error

HF_MODEL_NAME=${HF_USER}/Qwen2.5-7B_mis_cliph2.0_step400
HF_REPO_ID=$HF_MODEL_NAME  # HuggingFace repository name
# HF_MODEL_PATH for loading config, uses model name
HF_MODEL_PATH=Qwen/Qwen2.5-7B
# HF_MODEL_DIR is local model path, for copying tokenizer files (if present)
HF_MODEL_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
CHECKPOINT_DIR=${CHECKPOINT_PATH:-./checkpoints}/simpletir_last1500_maxres8000_maxpro16000_maxturn5_batch128_ppomini32_lossmodevanilla_maskvoidturnsTrue_risTrue_islvlsequence_isth5.0_simplelr_math_35_train_deepscaler_train_Qwen2.5-7B
STEP=400
TARGET_DIR=${CHECKPOINT_PATH:-./checkpoints}/output_models/Qwen2.5-7B_mis_cliph2.0_step400  # Local output path
# Whether to upload to HuggingFace (set to true to upload)
UPLOAD_TO_HF=true

# Create output directory
mkdir -p $TARGET_DIR

# Check if checkpoint directory exists
CHECKPOINT_PATH=$CHECKPOINT_DIR/global_step_$STEP/actor
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory does not exist: $CHECKPOINT_PATH"
    exit 1
fi

python model_merger.py \
    --backend fsdp \
    --hf_model_path $HF_MODEL_PATH \
    --local_dir $CHECKPOINT_PATH \
    --target_dir $TARGET_DIR

# Check if Python script executed successfully
if [ $? -ne 0 ]; then
    echo "Error: Model conversion failed. Exiting."
    exit 1
fi

# Copy tokenizer-related files
echo "Copying tokenizer files from $HF_MODEL_DIR to $TARGET_DIR..."
cp $HF_MODEL_DIR/tokenizer.json $TARGET_DIR/ 2>/dev/null || echo "Warning: tokenizer.json not found"
cp $HF_MODEL_DIR/tokenizer_config.json $TARGET_DIR/ 2>/dev/null || echo "Warning: tokenizer_config.json not found"  
cp $HF_MODEL_DIR/vocab.json $TARGET_DIR/ 2>/dev/null || echo "Warning: vocab.json not found"
cp $HF_MODEL_DIR/merges.txt $TARGET_DIR/ 2>/dev/null || echo "Warning: merges.txt not found"
cp $HF_MODEL_DIR/config.json $TARGET_DIR/ 2>/dev/null || echo "Warning: config.json not found"
cp $HF_MODEL_DIR/generation_config.json $TARGET_DIR/ 2>/dev/null || echo "Warning: generation_config.json not found"
echo "Tokenizer files copy completed."

# Upload to HuggingFace
if [ "$UPLOAD_TO_HF" = true ]; then
    echo "Uploading model to HuggingFace..."
    huggingface-cli upload $HF_MODEL_NAME $TARGET_DIR --repo-type model
    if [ $? -eq 0 ]; then
        echo "Model uploaded to HuggingFace successfully."
    else
        echo "Error: Failed to upload model to HuggingFace."
        exit 1
    fi
fi