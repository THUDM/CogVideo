# CogVideoX diffusers Fine-tuning Guide

[中文阅读](./README_zh.md)

[日本語で読む](./README_ja.md)

This feature is not fully complete yet. If you want to check the fine-tuning for the SAT version, please
see [here](../sat/README_zh.md). The dataset format is different from this version.

## Hardware Requirements

+ CogVideoX-2B / 5B LoRA: 1 * A100 (5B need to use `--use_8bit_adam`)
+ CogVideoX-2B SFT:  8 * A100 (Working)
+ CogVideoX-5B-I2V is not supported yet.

## Install Dependencies

Since the related code has not been merged into the diffusers release, you need to base your fine-tuning on the
diffusers branch. Please follow the steps below to install dependencies:

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now in Main branch
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

```
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \  # Use accelerate to launch multi-GPU training with the config file accelerate_config_machine_single.yaml
  train_cogvideox_lora.py \  # Training script train_cogvideox_lora.py for LoRA fine-tuning on CogVideoX model
  --gradient_checkpointing \  # Enable gradient checkpointing to reduce memory usage
  --pretrained_model_name_or_path $MODEL_PATH \  # Path to the pretrained model, specified by $MODEL_PATH
  --cache_dir $CACHE_PATH \  # Cache directory for model files, specified by $CACHE_PATH
  --enable_tiling \  # Enable tiling technique to process videos in chunks, saving memory
  --enable_slicing \  # Enable slicing to further optimize memory by slicing inputs
  --instance_data_root $DATASET_PATH \  # Dataset path specified by $DATASET_PATH
  --caption_column prompts.txt \  # Specify the file prompts.txt for video descriptions used in training
  --video_column videos.txt \  # Specify the file videos.txt for video paths used in training
  --validation_prompt "" \  # Prompt used for generating validation videos during training
  --validation_prompt_separator ::: \  # Set ::: as the separator for validation prompts
  --num_validation_videos 1 \  # Generate 1 validation video per validation round
  --validation_epochs 100 \  # Perform validation every 100 training epochs
  --seed 42 \  # Set random seed to 42 for reproducibility
  --rank 128 \  # Set the rank for LoRA parameters to 128
  --lora_alpha 64 \  # Set the alpha parameter for LoRA to 64, adjusting LoRA learning rate
  --mixed_precision bf16 \  # Use bf16 mixed precision for training to save memory
  --output_dir $OUTPUT_PATH \  # Specify the output directory for the model, defined by $OUTPUT_PATH
  --height 480 \  # Set video height to 480 pixels
  --width 720 \  # Set video width to 720 pixels
  --fps 8 \  # Set video frame rate to 8 frames per second
  --max_num_frames 49 \  # Set the maximum number of frames per video to 49
  --skip_frames_start 0 \  # Skip 0 frames at the start of the video
  --skip_frames_end 0 \  # Skip 0 frames at the end of the video
  --train_batch_size 4 \  # Set training batch size to 4
  --num_train_epochs 30 \  # Total number of training epochs set to 30
  --checkpointing_steps 1000 \  # Save model checkpoint every 1000 steps
  --gradient_accumulation_steps 1 \  # Accumulate gradients for 1 step, updating after each batch
  --learning_rate 1e-3 \  # Set learning rate to 0.001
  --lr_scheduler cosine_with_restarts \  # Use cosine learning rate scheduler with restarts
  --lr_warmup_steps 200 \  # Warm up the learning rate for the first 200 steps
  --lr_num_cycles 1 \  # Set the number of learning rate cycles to 1
  --optimizer AdamW \  # Use the AdamW optimizer
  --adam_beta1 0.9 \  # Set Adam optimizer beta1 parameter to 0.9
  --adam_beta2 0.95 \  # Set Adam optimizer beta2 parameter to 0.95
  --max_grad_norm 1.0 \  # Set maximum gradient clipping value to 1.0
  --allow_tf32 \  # Enable TF32 to speed up training
  --report_to wandb  # Use Weights and Biases (wandb) for logging and monitoring the training
```

## Running the Script to Start Fine-tuning

Single Node (One GPU or Multi GPU) fine-tuning:

```shell
bash finetune_single_rank.sh
```

Multi-Node fine-tuning:

```shell
bash finetune_multi_rank.sh # Needs to be run on each node
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
