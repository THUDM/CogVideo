# CogVideoX Diffusers Fine-tuning Guide

[中文阅读](./README_zh.md)

[日本語で読む](./README_ja.md)

If you're looking for the fine-tuning instructions for the SAT version, please check [here](../sat/README_zh.md). The
dataset format for this version differs from the one used here.

🔥🔥 **News**: ```2025/03/24```: We have launched [CogKit](https://github.com/THUDM/CogKit), a fine-tuning and inference framework for the CogView4 and CogVideoX series. We recommend migrate to CogKit, as future fine-tuning work for the CogVideo series will primarily be maintained within CogKit.

## Hardware Requirements

| Model                      | Training Type  | Distribution Strategy                | Mixed Precision | Training Resolution (FxHxW) | Hardware Requirements   |
|----------------------------|----------------|--------------------------------------|-----------------|-----------------------------|-------------------------|
| cogvideox-t2v-2b           | lora (rank128) | DDP                                  | fp16            | 49x480x720                  | 16GB VRAM (NVIDIA 4080) |
| cogvideox-{t2v, i2v}-5b    | lora (rank128) | DDP                                  | bf16            | 49x480x720                  | 24GB VRAM (NVIDIA 4090) |
| cogvideox1.5-{t2v, i2v}-5b | lora (rank128) | DDP                                  | bf16            | 81x768x1360                 | 35GB VRAM (NVIDIA A100) |
| cogvideox-t2v-2b           | sft            | DDP                                  | fp16            | 49x480x720                  | 36GB VRAM (NVIDIA A100) |
| cogvideox-t2v-2b           | sft            | 1-GPU zero-2 + opt offload           | fp16            | 49x480x720                  | 17GB VRAM (NVIDIA 4090) |
| cogvideox-t2v-2b           | sft            | 8-GPU zero-2                         | fp16            | 49x480x720                  | 17GB VRAM (NVIDIA 4090) |
| cogvideox-t2v-2b           | sft            | 8-GPU zero-3                         | fp16            | 49x480x720                  | 19GB VRAM (NVIDIA 4090) |
| cogvideox-t2v-2b           | sft            | 8-GPU zero-3 + opt and param offload | bf16            | 49x480x720                  | 14GB VRAM (NVIDIA 4080) |
| cogvideox-{t2v, i2v}-5b    | sft            | 1-GPU zero-2 + opt offload           | bf16            | 49x480x720                  | 42GB VRAM (NVIDIA A100) |
| cogvideox-{t2v, i2v}-5b    | sft            | 8-GPU zero-2                         | bf16            | 49x480x720                  | 42GB VRAM (NVIDIA 4090) |
| cogvideox-{t2v, i2v}-5b    | sft            | 8-GPU zero-3                         | bf16            | 49x480x720                  | 43GB VRAM (NVIDIA 4090) |
| cogvideox-{t2v, i2v}-5b    | sft            | 8-GPU zero-3 + opt and param offload | bf16            | 49x480x720                  | 28GB VRAM (NVIDIA 5090) |
| cogvideox1.5-{t2v, i2v}-5b | sft            | 1-GPU zero-2 + opt offload           | bf16            | 81x768x1360                 | 56GB VRAM (NVIDIA A100) |
| cogvideox1.5-{t2v, i2v}-5b | sft            | 8-GPU zero-2                         | bf16            | 81x768x1360                 | 55GB VRAM (NVIDIA A100) |
| cogvideox1.5-{t2v, i2v}-5b | sft            | 8-GPU zero-3                         | bf16            | 81x768x1360                 | 55GB VRAM (NVIDIA A100) |
| cogvideox1.5-{t2v, i2v}-5b | sft            | 8-GPU zero-3 + opt and param offload | bf16            | 81x768x1360                 | 40GB VRAM (NVIDIA A100) |

## Install Dependencies

Since the relevant code has not yet been merged into the official `diffusers` release, you need to fine-tune based on
the diffusers branch. Follow the steps below to install the dependencies:

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now on the Main branch
pip install -e .
```

## Prepare the Dataset

First, you need to prepare your dataset. Depending on your task type (T2V or I2V), the dataset format will vary
slightly:

```
.
├── prompts.txt
├── videos
├── videos.txt
├── images     # (Optional) For I2V, if not provided, first frame will be extracted from video as reference
└── images.txt # (Optional) For I2V, if not provided, first frame will be extracted from video as reference
```

Where:

- `prompts.txt`: Contains the prompts
- `videos/`: Contains the .mp4 video files
- `videos.txt`: Contains the list of video files in the `videos/` directory
- `images/`: (Optional) Contains the .png reference image files
- `images.txt`: (Optional) Contains the list of reference image files

You can download a sample dataset (
T2V) [Disney Steamboat Willie](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset).

If you need to use a validation dataset during training, make sure to provide a validation dataset with the same format
as the training dataset.

## Running Scripts to Start Fine-tuning

Before starting training, please note the following resolution requirements:

1. The number of frames must be a multiple of 8 **plus 1** (i.e., 8N+1), such as 49, 81 ...
2. Recommended video resolutions for each model:
    - CogVideoX: 480x720 (height x width)
    - CogVideoX1.5: 768x1360 (height x width)
3. For samples (videos or images) that don't match the training resolution, the code will directly resize them. This may
   cause aspect ratio distortion and affect training results. It's recommended to preprocess your samples (e.g., using
   crop + resize to maintain aspect ratio) before training.

> **Important Note**: To improve training efficiency, we automatically encode videos and cache the results on disk
> before training. If you modify the data after training, please delete the latent directory under the video directory to
> ensure the latest data is used.

### LoRA

```bash
# Modify configuration parameters in train_ddp_t2v.sh
# Main parameters to modify:
# --output_dir: Output directory
# --data_root: Dataset root directory
# --caption_column: Path to prompt file
# --image_column: Optional for I2V, path to reference image file list (remove this parameter to use the first frame of video as image condition)
# --video_column: Path to video file list
# --train_resolution: Training resolution (frames x height x width)
# For other important parameters, please refer to the launch script

bash train_ddp_t2v.sh  # Text-to-Video (T2V) fine-tuning
bash train_ddp_i2v.sh  # Image-to-Video (I2V) fine-tuning
```

### SFT

We provide several zero configuration templates in the `configs/` directory. Please choose the appropriate training
configuration based on your needs (configure the `deepspeed_config_file` option in `accelerate_config.yaml`).

```bash
# Parameters to configure are the same as LoRA training

bash train_zero_t2v.sh  # Text-to-Video (T2V) fine-tuning
bash train_zero_i2v.sh  # Image-to-Video (I2V) fine-tuning
```

In addition to setting the bash script parameters, you need to set the relevant training options in the zero
configuration file and ensure the zero training configuration matches the parameters in the bash script, such as
batch_size, gradient_accumulation_steps, mixed_precision. For details, please refer to
the [DeepSpeed official documentation](https://www.deepspeed.ai/docs/config-json/)

When using SFT training, please note:

1. For SFT training, model offload is not used during validation, so the peak VRAM usage may exceed 24GB. For GPUs with
   less than 24GB VRAM, it's recommended to disable validation.

2. Validation is slow when zero-3 is enabled, so it's recommended to disable validation when using zero-3.

## Load the Fine-tuned Model

+ Please refer to [cli_demo.py](../inference/cli_demo.py) for instructions on how to load the fine-tuned model.

+ For SFT trained models, please first use the `zero_to_fp32.py` script in the `checkpoint-*/` directory to merge the
  model weights

## Best Practices

+ We included 70 training videos with a resolution of `200 x 480 x 720` (frames x height x width). Through frame
  skipping in the data preprocessing, we created two smaller datasets with 49 and 16 frames to speed up experiments. The
  maximum frame count recommended by the CogVideoX team is 49 frames. These 70 videos were divided into three groups:
  10, 25, and 50 videos, with similar conceptual nature.
+ Videos with 25 or more frames work best for training new concepts and styles.
+ It's recommended to use an identifier token, which can be specified using `--id_token`, for better training results.
  This is similar to Dreambooth training, though regular fine-tuning without using this token will still work.
+ The original repository uses `lora_alpha` set to 1. We found that this value performed poorly in several runs,
  possibly due to differences in the model backend and training settings. Our recommendation is to set `lora_alpha` to
  be equal to the rank or `rank // 2`.
+ It's advised to use a rank of 64 or higher.
