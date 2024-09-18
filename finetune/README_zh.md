# CogVideoX diffusers 微调方案

[Read this in English](./README_zh.md)

[日本語で読む](./README_ja.md)

本功能尚未完全完善，如果您想查看SAT版本微调，请查看[这里](../sat/README_zh.md)。其数据集格式与本版本不同。

## 硬件要求

+ CogVideoX-2B LORA: 1 * A100
+ CogVideoX-2B SFT:  8 * A100
+ CogVideoX-5B/5B-I2V 暂未支持

## 安装依赖

由于相关代码还没有被合并到diffusers发行版，你需要基于diffusers分支进行微调。请按照以下步骤安装依赖：

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout cogvideox-lora-and-training
pip install -e .
```

## 准备数据集

首先，你需要准备数据集，数据集格式如下，其中，videos.txt 存放 videos 中的视频。

```
.
├── prompts.txt
├── videos
└── videos.txt
```

你可以从这里下载 [迪士尼汽船威利号](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset)

视频微调数据集作为测试微调。

## 配置文件和运行

`accelerate` 配置文件如下:

+ accelerate_config_machine_multi.yaml 适合多GPU使用
+ accelerate_config_machine_single.yaml 适合单GPU使用

`finetune` 脚本配置文件如下:

```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  
# 这条命令设置了 PyTorch 的 CUDA 内存分配策略，将显存扩展为段式内存管理，以防止 OOM（Out of Memory）错误。

accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
# 使用 Accelerate 启动训练，指定配置文件 `accelerate_config_machine_single.yaml`，并使用多 GPU。

  train_cogvideox_lora.py \
  # 这是你要执行的训练脚本，用于 LoRA 微调 CogVideoX 模型。

  --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  # 预训练模型的路径，指向你要微调的 CogVideoX-5b 模型。

  --cache_dir ~/.cache \
  # 模型缓存的目录，用于存储从 Hugging Face 下载的模型和数据集。

  --enable_tiling \
  # 启用 VAE tiling 功能，通过将图像划分成更小的区块处理，减少显存占用。

  --enable_slicing \
  # 启用 VAE slicing 功能，将图像在通道上切片处理，以节省显存。

  --instance_data_root ~/disney/ \
  # 实例数据的根目录，训练时使用的数据集文件夹。

  --caption_column prompts.txt \
  # 用于指定包含实例提示（文本描述）的列或文件，在本例中为 `prompts.txt` 文件。

  --video_column videos.txt \
  # 用于指定包含视频路径的列或文件，在本例中为 `videos.txt` 文件。

  --validation_prompt "Mickey with the captain and friends:::Mickey and the bear" \
  # 用于验证的提示语，多个提示语用指定分隔符（例如 `:::`）分开。

  --validation_prompt_separator ::: \
  # 验证提示语的分隔符，在此设置为 `:::`。

  --num_validation_videos 1 \
  # 验证期间生成的视频数量，设置为 1。

  --validation_epochs 2 \
  # 每隔多少个 epoch 运行一次验证，设置为每 2 个 epoch 验证一次。

  --seed 3407 \
  # 设置随机数种子，确保训练的可重复性，设置为 3407。

  --rank 128 \
  # LoRA 更新矩阵的维度，控制 LoRA 层的参数大小，设置为 128。

  --mixed_precision bf16 \
  # 使用混合精度训练，设置为 `bf16`（bfloat16），可以减少显存占用并加速训练。

  --output_dir cogvideox-lora-single-gpu \
  # 输出目录，存放模型预测结果和检查点。

  --height 480 \
  # 输入视频的高度，所有视频将被调整到 480 像素。

  --width 720 \
  # 输入视频的宽度，所有视频将被调整到 720 像素。

  --fps 8 \
  # 输入视频的帧率，所有视频将以每秒 8 帧处理。

  --max_num_frames 49 \
  # 输入视频的最大帧数，视频将被截取到最多 49 帧。

  --skip_frames_start 0 \
  # 每个视频从头部开始跳过的帧数，设置为 0，表示不跳过帧。

  --skip_frames_end 0 \
  # 每个视频从尾部跳过的帧数，设置为 0，表示不跳过尾帧。

  --train_batch_size 1 \
  # 训练的批次大小，每个设备的训练批次设置为 1。

  --num_train_epochs 10 \
  # 训练的总 epoch 数，设置为 10。

  --checkpointing_steps 500 \
  # 每经过 500 步保存一次检查点。

  --gradient_accumulation_steps 1 \
  # 梯度累积步数，表示每进行 1 步才进行一次梯度更新。

  --learning_rate 1e-4 \
  # 初始学习率，设置为 1e-4。

  --optimizer AdamW \
  # 优化器类型，选择 AdamW 优化器。

  --adam_beta1 0.9 \
  # Adam 优化器的 beta1 参数，设置为 0.9。

  --adam_beta2 0.95 \
  # Adam 优化器的 beta2 参数，设置为 0.95。
```

## 运行脚本，开始微调

单卡微调：

```shell
bash finetune_single_gpu.sh
```

多卡微调：

```shell
bash finetune_multi_gpus_1.sh #需要在每个节点运行
```

## 载入微调的模型

+ 请关注[cli_demo.py](../inference/cli_demo.py) 以了解如何加载微调的模型。

## 最佳实践

+ 包含70个分辨率为 `200 x 480 x 720`（帧数 x 高 x
  宽）的训练视频。通过数据预处理中的帧跳过，我们创建了两个较小的49帧和16帧数据集，以加快实验速度，因为CogVideoX团队建议的最大帧数限制是49帧。我们将70个视频分成三组，分别为10、25和50个视频。这些视频的概念性质相似。
+ 25个及以上的视频在训练新概念和风格时效果最佳。
+ 现使用可以通过 `--id_token` 指定的标识符token进行训练效果更好。这类似于 Dreambooth 训练，但不使用这种token的常规微调也可以工作。
+ 原始仓库使用 `lora_alpha` 设置为 1。我们发现这个值在多次运行中效果不佳，可能是因为模型后端和训练设置的不同。我们的建议是将
  lora_alpha 设置为与 rank 相同或 rank // 2。
+ 建议使用 rank 为 64 及以上的设置。

