#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

OPTIONS_NCCL="NCCL_DEBUG=warning NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
# HOST_FILE_PATH="hostfile_single"

video_data_test="" # TODO
CHECKPOINT_PATH="" # TODO: CogView2 ckpt

config_json="$script_dir/ds_config_zero.json"
gpt_options=" \
       --experiment-name pretrain-cogvideo-stage2 \
       --tokenizer-type fake \
       --vocab-size 150010 \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --num-workers 0 \
       --num-layers 48 \
       --hidden-size 3072 \
       --num-attention-heads 48 \
       --layout 64,464,2064 \
       --window-size 10 \
       --cogvideo-stage 2 \
       --additional-seqlen 2000 \
       --train-iters 500000 \
       --resume-dataloader \
       --train-data ${video_data_test}  \
       --train-data-weights 1 \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .001 \
       --checkpoint-activations \
       --max-sequence-length 1024 \
       --fp16 \
       --save-interval 2000 \
       --eval-interval 500 \
       --eval-iters 15 \
       --log-interval 50 \
       --save $main_dir/checkpoints \
       --sandwich-ln \
       --load $CHECKPOINT_PATH \
"
       # --load $CHECKPOINT_PATH \
       #  \       --sandwich-ln


gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"

#!/bin/bash

# Distribute Example
#export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
#export NCCL_IB_CUDA_SUPPORT=1
#export NCCL_IB_GID_INDEX=3
#export NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
export NCCL_DEBUG=info
export OMP_NUM_THREADS=4

if [ $RLAUNCH_REPLICA == "0" ]; then
	ifconfig eth0 | grep inet | grep -v inet6 | awk '{print $2}' > master_ip
fi

function finish {
	rm -rf master_ip
}

trap finish EXIT INT TERM

while [ ! -f master_ip ]; do
	echo "wait master_ip..."
	ls > /dev/null && sleep 1;
done

export MASTER_ADDR=$(cat master_ip)
echo "master_ip: $MASTER_ADDR"

MP_SIZE=1
task_set=$2
source $1
DATESTR=$(date +"%m-%d-%H-%M")

mkdir logs
run_cmd="sudo /opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=8 \
       	--nnodes=$RLAUNCH_REPLICA_TOTAL --node_rank=$RLAUNCH_REPLICA \
	--master_addr=$MASTER_ADDR --master_port=12355 pretrain_cogvideo.py $@ ${gpt_options}  2>&1 | tee logs/log-${DATESTR}-${RLAUNCH_REPLICA}.txt"
              

# run_cmd="${OPTIONS_NCCL} deepspeed  --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_video_swin_cond_glm_interp.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
