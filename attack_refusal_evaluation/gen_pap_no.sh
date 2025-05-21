#!/bin/bash
model_paths=(
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
)
out_names=(
    r1_distill_qwen_7b
)
for i in ${!model_paths[@]}; do
    model_path=${model_paths[$i]}
    out_name=${out_names[$i]}
    for test_name in pap_advbenchsub raw_advbenchsub; do

    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1 python run_gen_eval.py --model_path ${model_path} --model_name qwen --input_file results/${test_name}.jsonl --output_file score_results/${out_name}-${test_name}.json --limit 0 --gen 1 --system_prompt "" --use_vllm true

    done
done
