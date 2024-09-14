#! /bin/bash

echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_5b_i2v_lora.yaml configs/sft.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"