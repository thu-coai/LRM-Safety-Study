#!/bin/bash

# Define model checkpoints and their dedicated GPUs (path GPU_ID)
# Update this array with your desired paths and GPU IDs
MODEL_CHECKPOINTS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 7"
)

OUTPUT_DIR="./configs/auto"

# Prepare array of just model paths for generate_configs.py
MODEL_PATHS=()
for ckpt_entry in "${MODEL_CHECKPOINTS[@]}"; do
    read -r ckpt_path gpu_id <<< "$ckpt_entry"
    MODEL_PATHS+=("$ckpt_path")
done

#the python script to generate config files
python generate_configs.py \
    --template_path "./configs/template.yaml" \
    --output_dir "$OUTPUT_DIR" \
    --model_checkpoints "${MODEL_PATHS[@]}"

# Run run_pair.py for each generated config
for ckpt_entry in "${MODEL_CHECKPOINTS[@]}"; do
    # Split the entry into path and gpu_id
    read -r ckpt_path gpu_id <<< "$ckpt_entry"

    # Extract the directory containing the checkpoint or the path itself if not a checkpoint file
    filename=$(basename "$ckpt_path")
    if [[ "$filename" == *"checkpoint"* ]]; then
        ckpt_dir=$(dirname "$ckpt_path")
    else
        ckpt_dir="$ckpt_path"
    fi
    model_name=$(basename "$ckpt_dir")
    config_file="$OUTPUT_DIR/pair_${model_name}.yaml"

    if [ -f "$config_file" ]; then
        # Get the session name from the config filename (remove extension)
        session_name=$(basename "$config_file" .yaml)
        echo "$session_name on GPU $gpu_id"
        full_command=". ~/.bashrc && conda activate deploy && CUDA_VISIBLE_DEVICES=$gpu_id python run_pair.py --config_path \"$config_file\" && exit"

        # Create a new detached tmux session and run the command
        tmux new-session -d -s "$session_name" "$full_command"

    else
        echo "Warning: Generated config file not found for checkpoint $ckpt_path: $config_file"
    fi
done
