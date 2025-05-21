#!/bin/bash

train_data_path="../data/sft/sft_train_template_safety1k_math4k.json"
train_model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
save_path="../save/sft/r1_distill_qwen_7b_template_safety1k_math4k_max1_lr5e-6"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 --main_process_port 29522 \
    train_sft.py \
    --dataset_path $train_data_path \
    --model_path $train_model_path \
    --per_device_train_batch_size=2 \
    --num_train_epochs=1 \
    --logging_steps=10 \
    --output_dir=$save_path \
    --overwrite_output_dir=true \
    --bf16=true \
    --deepspeed=deepspeed_zero2.json \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=true \
    --max_grad_norm 1.0 \
    --learning_rate=5e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 0 \
    --max_seq_length=4096 \
    --save_only_model=true \
    --report_to=tensorboard \
    --save_strategy=epoch \
    --save_steps=300 \
    --seed=42 \
    --save_total_limit=0 2>&1 | tee logs/sfttrain_log.txt 
