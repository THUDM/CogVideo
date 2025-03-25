set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Select the model type
# The model is downloaded to a specified location on disk,
# or you can simply use the model's ID on Hugging Face,
# which will then be downloaded to the default cache path on Hugging Face.

export MODEL_TYPE="CogVideoX"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["CogVideoX"]="parallel_inference_xdit.py /cfs/dit/CogVideoX-2b 20"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results

# task args
if [ "$MODEL_TYPE" = "CogVideoX" ]; then
  TASK_ARGS="--height 480 --width 720 --num_frames 9"
fi

# CogVideoX asserts sp_degree == ulysses_degree*ring_degree <= 2. Also, do not set the pipefusion degree.
if [ "$MODEL_TYPE" = "CogVideoX" ]; then
N_GPUS=4
PARALLEL_ARGS="--ulysses_degree 2 --ring_degree 1"
CFG_ARGS="--use_cfg_parallel"
fi


torchrun --nproc_per_node=$N_GPUS ./$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "A small dog." \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG
