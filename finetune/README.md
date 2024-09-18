# CogVideoX diffusers Fine-tuning Guide

[中文阅读](./README_zh.md)

[日本語で読む](./README_ja.md)

This feature is not fully complete yet. If you want to check the fine-tuning for the SAT version, please
see [here](../sat/README_zh.md). The dataset format is different from this version.

## Hardware Requirements

+ CogVideoX-2B LoRA: 1 * A100
+ CogVideoX-2B SFT:  8 * A100
+ CogVideoX-5B/5B-I2V is not supported yet.

## Install Dependencies

Since the related code has not been merged into the diffusers release, you need to base your fine-tuning on the
diffusers branch. Please follow the steps below to install dependencies:

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout cogvideox-lora-and-training
pip install -e .
```

## Prepare the Dataset

First, you need to prepare the dataset. The dataset format should be as follows, with `videos.txt` containing the list
of videos in the `videos` directory:

```
.
├── prompts.txt
├── videos
└── videos.txt
```

You can download
the [Disney Steamboat Willie](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) dataset from
here.

This video fine-tuning dataset is used as a test for fine-tuning.

## Configuration Files and Execution

The `accelerate` configuration files are as follows:

+ `accelerate_config_machine_multi.yaml`: Suitable for multi-GPU use
+ `accelerate_config_machine_single.yaml`: Suitable for single-GPU use

The configuration for the `finetune` script is as follows:

```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  
# This command sets the PyTorch CUDA memory allocation strategy to expandable segments to prevent OOM (Out of Memory) errors.

accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu # Launch training using Accelerate with the specified config file for multi-GPU.

  train_cogvideox_lora.py   # This is the training script for LoRA fine-tuning of the CogVideoX model.

  --pretrained_model_name_or_path THUDM/CogVideoX-2b   # Path to the pretrained model you want to fine-tune, pointing to the CogVideoX-2b model.

  --cache_dir ~/.cache   # Directory for caching models downloaded from Hugging Face.

  --enable_tiling   # Enable VAE tiling to reduce memory usage by processing images in smaller chunks.

  --enable_slicing   # Enable VAE slicing to split the image into slices along the channel to save memory.

  --instance_data_root ~/disney/   # Root directory for instance data, i.e., the dataset used for training.

  --caption_column prompts.txt   # Specify the column or file containing instance prompts (text descriptions), in this case, the `prompts.txt` file.

  --video_column videos.txt   # Specify the column or file containing video paths, in this case, the `videos.txt` file.

  --validation_prompt "Mickey with the captain and friends:::Mickey and the bear"   # Validation prompts; multiple prompts are separated by the specified delimiter (e.g., `:::`).

  --validation_prompt_separator :::   # The separator for validation prompts, set to `:::` here.

  --num_validation_videos 1   # Number of videos to generate during validation, set to 1.

  --validation_epochs 2   # Number of epochs after which validation will be run, set to every 2 epochs.

  --seed 3407   # Set a random seed to ensure reproducibility, set to 3407.

  --rank 128   # Dimension of the LoRA update matrix, controls the size of the LoRA layers, set to 128.

  --mixed_precision bf16   # Use mixed precision training, set to `bf16` (bfloat16) to reduce memory usage and speed up training.

  --output_dir cogvideox-lora-single-gpu   # Output directory for storing model predictions and checkpoints.

  --height 480   # Height of the input videos, all videos will be resized to 480 pixels.

  --width 720   # Width of the input videos, all videos will be resized to 720 pixels.

  --fps 8   # Frame rate of the input videos, all videos will be processed at 8 frames per second.

  --max_num_frames 49   # Maximum number of frames per input video, videos will be truncated to 49 frames.

  --skip_frames_start 0   # Number of frames to skip from the start of each video, set to 0 to not skip any frames.

  --skip_frames_end 0   # Number of frames to skip from the end of each video, set to 0 to not skip any frames.

  --train_batch_size 1   # Training batch size per device, set to 1.

  --num_train_epochs 10   # Total number of training epochs, set to 10.

  --checkpointing_steps 500   # Save checkpoints every 500 steps.

  --gradient_accumulation_steps 1   # Gradient accumulation steps, perform an update every 1 step.

  --learning_rate 1e-4   # Initial learning rate, set to 1e-4.

  --optimizer AdamW   # Optimizer type, using AdamW optimizer.

  --adam_beta1 0.9   # Beta1 parameter for the Adam optimizer, set to 0.9.

  --adam_beta2 0.95   # Beta2 parameter for the Adam optimizer, set to 0.95.
```

## Running the Script to Start Fine-tuning

Single GPU fine-tuning:

```shell
bash finetune_single_gpu.sh
```

Multi-GPU fine-tuning:

```shell
bash finetune_multi_gpus_1.sh # Needs to be run on each node
```

## Loading the Fine-tuned Model

+ Please refer to [cli_demo.py](../inference/cli_demo.py) for how to load the fine-tuned model.

## Best Practices

+ Includes 70 training videos with a resolution of `200 x 480 x 720` (frames x height x width). By skipping frames in
  the data preprocessing, we created two smaller datasets with 49 and 16 frames to speed up experimentation, as the
  maximum frame limit recommended by the CogVideoX team is 49 frames. We split the 70 videos into three groups of 10,
  25, and 50 videos, with similar conceptual nature.
+ Using 25 or more videos works best when training new concepts and styles.
+ It works better to train using identifier tokens specified with `--id_token`. This is similar to Dreambooth training,
  but regular fine-tuning without such tokens also works.
+ The original repository used `lora_alpha` set to 1. We found this value ineffective across multiple runs, likely due
  to differences in the backend and training setup. Our recommendation is to set `lora_alpha` equal to rank or rank //
  2.
+ We recommend using a rank of 64 or higher.
