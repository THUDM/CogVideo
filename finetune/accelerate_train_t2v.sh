#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B"
    --model_name "cogvideox1.5-t2v"
    --model_type "t2v"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/path/to/output/dir"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/path/to/data/dir"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
    --train_resolution "80x768x1360"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
    --seed 42
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 200
    --checkpointing_limit 10
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation False
    --validation_dir "/path/to/validation/dir"
    --validation_steps 400
    --validation_prompts "prompts.txt"
    --gen_fps 15
)

# Combine all arguments and launch training
accelerate launch train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"