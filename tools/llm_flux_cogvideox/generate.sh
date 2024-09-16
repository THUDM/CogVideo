#!/bin/bash

NUM_VIDEOS=100
INFERENCE_STEPS=50
GUIDANCE_SCALE=7.0
OUTPUT_DIR_PREFIX="outputs/gpu_"
LOG_DIR_PREFIX="logs/gpu_"

CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_DEVICES"

for i in "${!GPU_ARRAY[@]}"
do
    GPU=${GPU_ARRAY[$i]}
    echo "Starting task on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 llm_flux_cogvideox.py \
    --num_videos $NUM_VIDEOS \
    --image_generator_num_inference_steps $INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --use_dynamic_cfg \
    --output_dir ${OUTPUT_DIR_PREFIX}${GPU} \
    > ${LOG_DIR_PREFIX}${GPU}.log 2>&1 &
done