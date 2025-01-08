# CogVideoX Diffusers Fine-tuning Guide

[中文阅读](./README_zh.md)

[日本語で読む](./README_ja.md)

If you're looking for the fine-tuning instructions for the SAT version, please check [here](../sat/README_zh.md). The dataset format for this version differs from the one used here.

## Hardware Requirements

| Model                | Training Type   | Mixed Precision | Training Resolution (frames x height x width) | Hardware Requirements    |
|---------------------|-----------------|----------------|---------------------------------------------|------------------------|
| cogvideox-t2v-2b     | lora (rank128)  | fp16           | 49x480x720                                  | 16GB VRAM (NVIDIA 4080) |
| cogvideox-t2v-5b     | lora (rank128)  | bf16           | 49x480x720                                  | 24GB VRAM (NVIDIA 4090) |
| cogvideox-i2v-5b     | lora (rank128)  | bf16           | 49x480x720                                  | 24GB VRAM (NVIDIA 4090) |
| cogvideox1.5-t2v-5b  | lora (rank128)  | bf16           | 81x768x1360                                 | 35GB VRAM (NVIDIA A100) |
| cogvideox1.5-i2v-5b  | lora (rank128)  | bf16           | 81x768x1360                                 | 35GB VRAM (NVIDIA A100) |


## Install Dependencies

Since the relevant code has not yet been merged into the official `diffusers` release, you need to fine-tune based on the diffusers branch. Follow the steps below to install the dependencies:

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now on the Main branch
pip install -e .
```

## Prepare the Dataset

First, you need to prepare your dataset. Depending on your task type (T2V or I2V), the dataset format will vary slightly:

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

You can download a sample dataset (T2V) [Disney Steamboat Willie](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset).

If you need to use a validation dataset during training, make sure to provide a validation dataset with the same format as the training dataset.

## Run the Script to Start Fine-tuning

Before starting the training, please note the following resolution requirements:

1. The number of frames must be a multiple of 8 **plus 1** (i.e., 8N+1), such as 49, 81, etc.
2. The recommended resolution for videos is:
   - CogVideoX: 480x720 (Height x Width)
   - CogVideoX1.5: 768x1360 (Height x Width)
3. For samples that do not meet the required resolution (videos or images), the code will automatically resize them. This may distort the aspect ratio and impact training results. We recommend preprocessing the samples (e.g., using crop + resize to maintain aspect ratio) before training.

> **Important Note**: To improve training efficiency, we will automatically encode videos and cache the results on disk. If you modify the data after training has begun, please delete the `latent` directory under the `videos/` folder to ensure that the latest data is used.

### Text-to-Video (T2V) Fine-tuning

```bash
# Modify the configuration parameters in accelerate_train_t2v.sh
# The main parameters to modify are:
# --output_dir: Output directory
# --data_root: Root directory of the dataset
# --caption_column: Path to the prompt file
# --video_column: Path to the video list file
# --train_resolution: Training resolution (frames x height x width)
# Refer to the start script for other important parameters

bash accelerate_train_t2v.sh
```

### Image-to-Video (I2V) Fine-tuning

```bash
# Modify the configuration parameters in accelerate_train_i2v.sh
# In addition to modifying the same parameters as for T2V, you also need to set:
# --image_column: Path to the reference image list file(if not provided, remove use this parameter)
# Refer to the start script for other important parameters

bash accelerate_train_i2v.sh
```

## Load the Fine-tuned Model

+ Please refer to [cli_demo.py](../inference/cli_demo.py) for instructions on how to load the fine-tuned model.

## Best Practices

+ We included 70 training videos with a resolution of `200 x 480 x 720` (frames x height x width). Through frame skipping in the data preprocessing, we created two smaller datasets with 49 and 16 frames to speed up experiments. The maximum frame count recommended by the CogVideoX team is 49 frames. These 70 videos were divided into three groups: 10, 25, and 50 videos, with similar conceptual nature.
+ Videos with 25 or more frames work best for training new concepts and styles.
+ It's recommended to use an identifier token, which can be specified using `--id_token`, for better training results. This is similar to Dreambooth training, though regular fine-tuning without using this token will still work.
+ The original repository uses `lora_alpha` set to 1. We found that this value performed poorly in several runs, possibly due to differences in the model backend and training settings. Our recommendation is to set `lora_alpha` to be equal to the rank or `rank // 2`.
+ It's advised to use a rank of 64 or higher.