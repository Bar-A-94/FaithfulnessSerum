#!/bin/bash
# Run open-source model with PE-LRP attention intervention

python -u scripts/lrp_guidance.py \
    --dataset_specific_name all \
    --num_of_examples 2000 \
    --eval_model_path Qwen/Qwen2.5-14B-Instruct \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --alpha 0.06 \
    --verbose \
    --dataset commonsenseqa \
    --hint_type sycophancy
