# CogVideoX diffusers Fine-tuning Guide

If you want to see the SAT version fine-tuning, please check [here](../sat/README.md). The dataset format is different
from this version.

This tutorial aims to quickly fine-tune the diffusers version of the CogVideoX model.

### Hardware Requirements

+ CogVideoX-2B LORA: 1 * A100
+ CogVideoX-2B SFT:  8 * A100
+ CogVideoX-5B/5B-I2V not yet supported

### Prepare the Dataset

First, you need to prepare the dataset. The format of the dataset is as follows, where `videos.txt` contains paths to
the videos in the `videos` directory.

```
.
├── prompts.txt
├── videos
└── videos.txt
```

You can download [Disney Steamboat Willie](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset)
from here.

The video fine-tuning dataset is used as a test for fine-tuning.

### Configuration Files and Execution

`accelerate` configuration files are as follows:

+ accelerate_config_machine_multi.yaml for multi-GPU use
+ accelerate_config_machine_single.yaml for single-GPU use

The `finetune` script configuration is as follows:

```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# This command sets PyTorch's CUDA memory allocation strategy to segment-based memory management to prevent OOM (Out of Memory) errors.

accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \

# Use Accelerate to start training, specifying the `accelerate_config_machine_single.yaml` configuration file, and using multiple GPUs.

train_cogvideox_lora.py \

# This is the training script you will execute for LoRA fine-tuning of the CogVideoX model.

--pretrained_model_name_or_path THUDM/CogVideoX-2b \

# The path to the pretrained model, pointing to the CogVideoX-5b model you want to fine-tune.

--cache_dir ~/.cache \

# The directory where downloaded models and datasets will be stored.

--enable_tiling \

# Enable VAE tiling functionality, which reduces memory usage by processing smaller blocks of the image.

--enable_slicing \

# Enable VAE slicing functionality, which slices the image across channels to save memory.

--instance_data_root ~/disney/ \

# The root directory of the instance data, the folder of the dataset used during training.

--caption_column prompts.txt \

# Specifies the column or file containing instance prompts (text descriptions), in this case, the `prompts.txt` file.

--video_column videos.txt \

# Specifies the column or file containing paths to videos, in this case, the `videos.txt` file.

--validation_prompt "Mickey with the captain and friends:::Mickey and the bear" \

# The prompt(s) used for validation, multiple prompts should be separated by the specified delimiter (`:::`).

--validation_prompt_separator ::: \

# The delimiter for validation prompts, set here as `:::`.

--num_validation_videos 1 \

# The number of videos to be generated during validation, set to 1.

--validation_epochs 2 \

# How many epochs to run validation, set to validate every 2 epochs.

--seed 3407 \

# Sets the random seed for reproducible training, set to 3407.

--rank 128 \

# The dimension of the LoRA update matrices, controlling the size of the LoRA layer parameters, set to 128.

--mixed_precision bf16 \

# Use mixed precision training, set to `bf16` (bfloat16), which can reduce memory usage and speed up training.

--output_dir cogvideox-lora-single-gpu \

# Output directory, where model predictions and checkpoints will be stored.

--height 480 \

# The height of input videos, all videos will be resized to 480 pixels.

--width 720 \

# The width of input videos, all videos will be resized to 720 pixels.

--fps 8 \

# The frame rate of input videos, all videos will be processed at 8 frames per second.

--max_num_frames 49 \

# The maximum number of frames for input videos, videos will be truncated to a maximum of 49 frames.

--skip_frames_start 0 \

# The number of frames to skip at the beginning of each video, set to 0, indicating no frames are skipped.

--skip_frames_end 0 \

# The number of frames to skip at the end of each video, set to 0, indicating no frames are skipped.

--train_batch_size 1 \

# The batch size for training, set to 1 per device.

--num_train_epochs 10 \

# The total number of epochs for training, set to 10.

--checkpointing_steps 500 \

# Save a checkpoint every 500 steps.

--gradient_accumulation_steps 1 \

# The number of gradient accumulation steps, indicating that a gradient update is performed every 1 step.

--learning_rate 1e-4 \

# The initial learning rate, set to 1e-4.

--optimizer AdamW \

# The type of optimizer, choosing AdamW.

--adam_beta1 0.9 \

# The beta1 parameter for the Adam optimizer, set to 0.9.

--adam_beta2 0.95 \

# The beta2 parameter for the Adam optimizer, set to 0.95.

```

### Run the script to start fine-tuning

Single GPU fine-tuning:

```shell
bash finetune_single_gpu.sh
```

Multi-GPU fine-tuning:

```shell
bash finetune_multi_gpus_1.sh # needs to be run on each node
```

### Best Practices

+ Include 70 videos with a resolution of `200 x 480 x 720` (frames x height x width). Through data preprocessing's frame
  skipping, we created two smaller datasets of 49 and 16 frames to speed up experiments, as the CogVideoX team suggests
  a maximum frame count of 49. We divided the 70 videos into three groups of 10, 25, and 50 videos. These videos are
  conceptually similar.
+ 25 or more videos work best when training new concepts and styles.
+ Now using an identifier token specified through `--id_token` enhances training results. This is similar to Dreambooth
  training, but regular fine-tuning without this token also works.
+ The original repository uses `lora_alpha` set to 1. We found this value to be ineffective in multiple runs, likely due
  to differences in model backend and training setups. Our recommendation is to set lora_alpha to the same as rank or
  rank // 2.
+ Using settings with a rank of 64 or above is recommended.