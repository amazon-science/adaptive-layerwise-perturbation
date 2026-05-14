#!/bin/bash

# Download all required datasets from HuggingFace

cd ${PROJECT_DIR:-.}/eval

echo "Downloading datasets from HuggingFace..."

python3 download_datasets.py \
  --datasets \
    "weqweasdas/math500" \
    "weqweasdas/minerva_math" \
    "weqweasdas/olympiadbench" \
    "${HF_USER}/hmmt0225" \
  --output_dir ./datasets \
  --split train

echo ""
echo "Dataset download complete!"
echo "Files saved to: ./datasets"
echo ""
echo "Directory structure:"
ls -lh ./datasets/weqweasdas/ 2>/dev/null || echo "weqweasdas directory not found"
ls -lh ./datasets/${HF_USER}/ 2>/dev/null || echo "${HF_USER} directory not found"

