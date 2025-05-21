#!/bin/bash

model_name_or_paths=(
    # Insert the model path here
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
)
for model_name_or_path in "${model_name_or_paths[@]}"; do
    echo "Running evaluation for model: $model_name_or_path"
    echo "start aime_24 evaluation"
    python eval.py \
        --model_name_or_path "$model_name_or_path" \
        --data_name "aime_24" \
        --prompt_type "r1" \
        --temperature 0.6 \
        --start_idx 0 \
        --end_idx -1 \
        --n_sampling 4 \
        --k 1 \
        --max_tokens 32768 \
        --seed 0 \
        --top_p 0.95 \
        --surround_with_messages  > logs/aime_24/${model_name_or_path}.log 2>&1
    echo "start math500 evaluation"
    python eval.py \
        --model_name_or_path "$model_name_or_path" \
        --data_name "math500" \
        --prompt_type "r1" \
        --temperature 0.6 \
        --start_idx 0 \
        --end_idx -1 \
        --n_sampling 1 \
        --k 1 \
        --max_tokens 32768 \
        --seed 0 \
        --top_p 0.95 \
        --surround_with_messages  > logs/math500/${model_name_or_path}.log 2>&1
done