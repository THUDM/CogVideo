#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B-I2V"
    --model_name "cogvideox1.5-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/absolute/path/to/your/output_dir"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/absolute/path/to/your/data_root"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "81x768x1360"  # (frames x height x width), frames should be 8N+1
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10
    --seed 42

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"]
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 5
    --checkpointing_limit 10
    --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false  # ["true", "false"]
    --validation_dir "/absolute/path/to/validation_set"
    --validation_steps 20  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
