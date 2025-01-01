#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Launch training with accelerate
accelerate launch train.py \
    ########## Model Configuration ##########
    --model_path "THUDM/CogVideoX1.5-5B" \
    --model_name "cogvideox1.5-t2v" \
    --model_type "t2v" \
    --training_type "lora" \
    
    ########## Output Configuration ##########
    --output_dir "/path/to/output/dir" \
    --report_to "tensorboard" \
    
    ########## Data Configuration ##########
    --data_root "/path/to/data/dir" \
    --caption_column "prompt.txt" \
    --video_column "videos.txt" \
    --train_resolution "48x768x1360" \
    
    ########## Training Configuration ##########
    --train_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision "bf16" \
    --seed 42 \
    
    ########## System Configuration ##########
    --num_workers 8 \
    --pin_memory True \
    --nccl_timeout 1800 \
    
    ########## Checkpointing Configuration ##########
    --checkpointing_steps 200 \
    --checkpointing_limit 10 \
    
    ########## Validation Configuration ##########
    --do_validation False \
    --validation_dir "path/to/validation/dir" \
    --validation_steps 400 \
    --validation_prompts "prompts.txt" \
    --gen_fps 15
