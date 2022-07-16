#!/bin/bash

NLAYERS=48
NHIDDEN=3072
NATT=48
MAXSEQLEN=1024
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MPSIZE=1

#SAMPLING ARGS
TEMP=1.05
TOPK=12

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

MASTER_PORT=${MASTER_PORT} SAT_HOME=/sharefs/cogview-new python cogvideo_pipeline.py \
        --input-source interactive \
        --output-path ./output \
        --parallel-size 1 \
        --both-stages \
        --use-guidance-stage1 \
        --guidance-alpha 3.0 \
        --generate-frame-num 5 \
        --tokenizer-type fake \
        --mode inference \
        --distributed-backend nccl \
        --fp16 \
        --model-parallel-size $MPSIZE \
        --temperature $TEMP \
        --coglm-temperature2 0.89 \
        --top_k $TOPK \
        --sandwich-ln \
        --seed 1234 \
        --num-workers 0 \
        --batch-size 4 \
        --max-inference-batch-size 8 \
        $@
