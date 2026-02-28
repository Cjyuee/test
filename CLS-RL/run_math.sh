#!/bin/bash

# Qwen/Qwen2-VL-2B-Instruct\
# dataset_name afdsafas/Caltech-4shot-b2n \
# --report_to none \ # wandb修改成none

# 记得修改grpo_math.py中210行附近的加载mathV360K数据集的路径信息
# --num_generations在2B模型的时候是8，7B模型的时候是4
cd src/cls-rl
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="debug_log_2b_instruct_math.txt"
# 增加了一行卡的设置
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export CUDA_VISIBLE_DEVICES=0
#Qwen/Qwen2-VL-2B-Instruct
PYTHONIOENCODING=utf-8 python -m torch.distributed.run --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="29501" \
    src/open_r1/grpo_math.py \
    --output_dir  'Qwen2-VL-2B-Instruct-math-generation4/'\
    --model_name_or_path  Qwen/Qwen2-VL-2B-Instruct\
    --dataset_name Zhiqiang007/MathV360 \
    --deepspeed /root/autodl-tmp/CLS-RL/src/cls-rl/local_scripts/zero3.json \
    --max_prompt_length 4096 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B-Instruct-math\
    --save_steps 10000 \
    --save_only_model true \
    --num_generations 8 \
    --temperature 1

