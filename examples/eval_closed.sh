#!/bin/bash
# Evaluate closed-source model outputs

python scripts/eval_closed.py \
    --dataset_specific_name all \
    --eval_model_path Qwen/Qwen2.5-14B-Instruct \
    --verbose \
    --eval_file_path <YOUR_GENERATION_FILE>
