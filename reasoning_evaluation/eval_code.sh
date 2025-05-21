#!/bin/bash
cd  LiveCodeBench

export port=8178
session="vllm_server"
GPU="0,1,2,3"
tensor_parallel_size=4
model_paths=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)
model_names=(
    "r1_distill_qwen_7b"
)
for i in ${!model_paths[@]}; do
    model_path=${model_paths[$i]}
    model_name=${model_names[$i]}
    echo "Start evaluation for model: $model_path"
    
    tmux new-session -d -s $session
    tmux send-keys -t $session "
    export CUDA_VISIBLE_DEVICES=$GPU
    model_path=${model_path}
    model=${model_name}
    python -m vllm.entrypoints.openai.api_server --model \$model_path --served-model-name \$model --trust-remote-code --port $port --gpu_memory_utilization 0.9 --tensor_parallel_size $tensor_parallel_size 
    "
    

    echo "vLLM server started in tmux session 'vllm_server'"
    echo "To attach to the session, run: tmux attach -t vllm_server"

    echo "Waiting 30 seconds for the server to initialize..."
    sleep 30

    worker=44
    PYTHONUNBUFFERED=1 python -m lcb_runner.runner.main --model ${model_name} --scenario codegeneration --evaluate --release_version release_v5 --start_date 2024-10-01 --max_tokens 30000 --n 1 --temperature 0.6 --top_p 0.95 --multiprocess $worker --openai_timeout 1000000000 --trust_remote_code

    echo "Finish evaluation for model: $model_path"
    #--enable_prefix_caching --use_cache --cache_batch_size 1

    # kill the tmux session
    tmux kill-session -t $session
    sleep 5

done
