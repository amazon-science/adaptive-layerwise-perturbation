#!/bin/bash

# Script to evaluate minerva_math for all models and checkpoints in gen_data/
# Usage: bash eval_all_minerva.sh [--force]
#
# Expects gen_data layout: <model>/global_step_<N>/merged/<author>/minerva_math/merged_data.jsonl
# (from eval_model_local.sh + gen_data on weqweasdas/minerva_math). Writes <model>/global_step_<N>/minerva_scores.txt.
# Skips steps that already have minerva_scores.txt unless --force.

# Configuration
# Uses gen_data layout: global_step_*/merged/<author>/minerva_math/merged_data.jsonl (not step_records/)
BASE_DIR="/home/chenluy/mismatch_perturbation-math_fp8/data/gen_data"
EVAL_BENCHMARK_DIR="/home/chenluy/mismatch_perturbation-math_fp8/eval_benchmark"
COMPUTE_SCORE_SCRIPT="$EVAL_BENCHMARK_DIR/compute_score_minerval.py"

# Parse arguments
FORCE_EVAL=false
if [ "$1" == "--force" ]; then
    FORCE_EVAL=true
    echo "Force mode enabled: will re-evaluate existing minerva_scores.txt files"
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory not found: $BASE_DIR"
    exit 1
fi

echo "=========================================="
echo "Auto-evaluating minerva_math samples"
echo "=========================================="
echo "Base directory: $BASE_DIR"
echo "Force mode: $FORCE_EVAL"
echo ""

# Find all model directories (directories that contain global_step_* subdirectories)
echo "Scanning for models..."
MODEL_DIRS=()
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        # Check if this directory contains any global_step_* subdirectories
        if ls "$dir"/global_step_* 1> /dev/null 2>&1; then
            MODEL_DIRS+=("$dir")
        fi
    fi
done

if [ ${#MODEL_DIRS[@]} -eq 0 ]; then
    echo "Error: No model directories found in $BASE_DIR"
    exit 1
fi

echo "Found ${#MODEL_DIRS[@]} model(s) to evaluate"
echo "=========================================="
echo ""

# Global counters
TOTAL_MODELS=0
TOTAL_STEPS_FOUND=0
TOTAL_PROCESSED=0
TOTAL_SKIPPED=0
TOTAL_ALREADY_EXISTS=0
TOTAL_SUCCESS=0

# Loop through each model
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    ((TOTAL_MODELS++))
    
    echo ""
    echo "######################################"
    echo "# Model: $MODEL_NAME"
    echo "######################################"
    echo ""
    
    # Find all global_step_* directories and sort them numerically
    STEP_DIRS=$(find "$MODEL_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V)
    
    if [ -z "$STEP_DIRS" ]; then
        echo "No global_step_* directories found, skipping model..."
        continue
    fi
    
    # Count total directories for this model
    NUM_STEPS=$(echo "$STEP_DIRS" | wc -l)
    ((TOTAL_STEPS_FOUND+=$NUM_STEPS))
    echo "Found $NUM_STEPS checkpoint(s)"
    echo ""
    
    # Loop through each step directory
    for STEP_DIR in $STEP_DIRS; do
        # Extract step number from directory name
        STEP_NUM=$(basename "$STEP_DIR" | sed 's/global_step_//')
        OUTPUT_FILE="$STEP_DIR/minerva_scores.txt"
        
        # Check if output file already exists
        if [ -f "$OUTPUT_FILE" ] && [ "$FORCE_EVAL" = false ]; then
            echo "⊘ Step $STEP_NUM: minerva_scores.txt already exists, skipping..."
            ((TOTAL_ALREADY_EXISTS++))
            continue
        fi
        
        # Find merged_data.jsonl under merged/*/minerva_math/ (gen_data layout)
        MERGED_DATA=""
        if [ -d "$STEP_DIR/merged" ]; then
            for candidate in "$STEP_DIR"/merged/*/minerva_math/merged_data.jsonl; do
                if [ -f "$candidate" ]; then
                    MERGED_DATA="$candidate"
                    break
                fi
            done
        fi
        
        if [ -z "$MERGED_DATA" ] || [ ! -f "$MERGED_DATA" ]; then
            echo "⊘ Step $STEP_NUM: merged/*/minerva_math/merged_data.jsonl not found, skipping..."
            ((TOTAL_SKIPPED++))
            continue
        fi
        
        echo "→ Processing step $STEP_NUM..."
        
        # Run compute_score_minerval.py (writes one line "path score" to OUTPUT_FILE)
        EVAL_OUTPUT=$(cd "$EVAL_BENCHMARK_DIR" && python3 "$COMPUTE_SCORE_SCRIPT" \
            --dataset_path "$MERGED_DATA" \
            --record_path "$OUTPUT_FILE" 2>&1)
        
        EVAL_EXIT_CODE=$?
        
        # Show key progress
        echo "$EVAL_OUTPUT" | grep -E "(Evaluate|Error)" || true
        
        if [ $EVAL_EXIT_CODE -ne 0 ]; then
            echo "  ✗ Step $STEP_NUM: Python script failed with exit code $EVAL_EXIT_CODE"
            echo "$EVAL_OUTPUT" | head -15
            ((TOTAL_PROCESSED++))
            continue
        fi
        
        if [ -f "$OUTPUT_FILE" ]; then
            # compute_score_minerval.py writes: "<dataset_path> <score>"
            AVG_SCORE=$(awk '{print $NF}' "$OUTPUT_FILE")
            if [ -n "$AVG_SCORE" ]; then
                echo "  ✓ Step $STEP_NUM: score = $AVG_SCORE"
                ((TOTAL_SUCCESS++))
            else
                echo "  ✗ Step $STEP_NUM: Failed to extract score from output file"
            fi
        else
            echo "  ✗ Step $STEP_NUM: Output file not created"
            echo "$EVAL_OUTPUT" | tail -5
        fi
        
        ((TOTAL_PROCESSED++))
    done
    
    echo ""
    echo "Model $MODEL_NAME completed"
    echo ""
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo "Summary:"
echo "  Total models: $TOTAL_MODELS"
echo "  Total checkpoints found: $TOTAL_STEPS_FOUND"
echo "  Already evaluated (skipped): $TOTAL_ALREADY_EXISTS"
echo "  Newly processed: $TOTAL_PROCESSED"
echo "  Successfully evaluated: $TOTAL_SUCCESS"
echo "  Skipped (missing files): $TOTAL_SKIPPED"
echo ""
echo "Results saved to:"
echo "  $BASE_DIR/*/global_step_*/minerva_scores.txt"
echo ""
if [ "$FORCE_EVAL" = false ]; then
    echo "Tip: Use --force to re-evaluate existing minerva_scores.txt files"
fi
echo "=========================================="

