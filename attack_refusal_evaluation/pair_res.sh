#!/bin/bash


gpu_id=6,7

models=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)



for model_path in "${models[@]}"; do

    filename=$(basename "$model_path")
    if [[ "$filename" == *"checkpoint"* ]]; then
        ckpt_dir=$(dirname "$model_path")
    else
        ckpt_dir="$model_path"
    fi
    model_name=$(basename "$ckpt_dir")

    echo "Preparing to test model: $model_name with path: $model_path"
    test_name=$model_name-pair_advbench
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu_id python run_gen_eval.py --model_path ${model_path}  --model_name ${model_name} --input_file results/pair/${test_name}.jsonl --output_file score_results/pair/${model_name}.json --limit 0 --gen 1 --do_sample true --max_new_tokens 2048 --score_key llamaguard3_score --use_vllm true > logs/pair/${model_name}.log 2>&1 --system_prompt "" 
done