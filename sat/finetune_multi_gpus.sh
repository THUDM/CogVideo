#! /bin/bash

echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_cmd="torchrun --standalone --nproc_per_node=4 train_video.py --base configs/cogvideox_2b_sft.yaml --seed $RANDOM“

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"