# CogVideoX diffusers 微调方案

[Read this in English](./README_zh.md)

[日本語で読む](./README_ja.md)

本功能尚未完全完善，如果您想查看SAT版本微调，请查看[这里](../sat/README_zh.md)。其数据集格式与本版本不同。

## 硬件要求

+ CogVideoX-2B / 5B T2V LORA: 1 * A100  (5B need to use `--use_8bit_adam`)
+ CogVideoX-2B SFT:  8 * A100 (制作中)
+ CogVideoX-5B-I2V 暂未支持

## 安装依赖

由于相关代码还没有被合并到diffusers发行版，你需要基于diffusers分支进行微调。请按照以下步骤安装依赖：

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now in Main branch
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

`finetune` 脚本配置文件如下：

```shell

accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \  # 使用 accelerate 启动多GPU训练，配置文件为 accelerate_config_machine_single.yaml
  train_cogvideox_lora.py \  # 运行的训练脚本为 train_cogvideox_lora.py，用于在 CogVideoX 模型上进行 LoRA 微调
  --gradient_checkpointing \  # 启用梯度检查点功能，以减少显存使用
  --pretrained_model_name_or_path $MODEL_PATH \  # 预训练模型路径，通过 $MODEL_PATH 指定
  --cache_dir $CACHE_PATH \  # 模型缓存路径，由 $CACHE_PATH 指定
  --enable_tiling \  # 启用tiling技术，以分片处理视频，节省显存
  --enable_slicing \  # 启用slicing技术，将输入切片，以进一步优化内存
  --instance_data_root $DATASET_PATH \  # 数据集路径，由 $DATASET_PATH 指定
  --caption_column prompts.txt \  # 指定用于训练的视频描述文件，文件名为 prompts.txt
  --video_column videos.txt \  # 指定用于训练的视频路径文件，文件名为 videos.txt
  --validation_prompt "" \  # 验证集的提示语 (prompt)，用于在训练期间生成验证视频
  --validation_prompt_separator ::: \  # 设置验证提示语的分隔符为 :::
  --num_validation_videos 1 \  # 每个验证回合生成 1 个视频
  --validation_epochs 100 \  # 每 100 个训练epoch进行一次验证
  --seed 42 \  # 设置随机种子为 42，以保证结果的可复现性
  --rank 128 \  # 设置 LoRA 参数的秩 (rank) 为 128
  --lora_alpha 64 \  # 设置 LoRA 的 alpha 参数为 64，用于调整LoRA的学习率
  --mixed_precision bf16 \  # 使用 bf16 混合精度进行训练，减少显存使用
  --output_dir $OUTPUT_PATH \  # 指定模型输出目录，由 $OUTPUT_PATH 定义
  --height 480 \  # 视频高度为 480 像素
  --width 720 \  # 视频宽度为 720 像素
  --fps 8 \  # 视频帧率设置为 8 帧每秒
  --max_num_frames 49 \  # 每个视频的最大帧数为 49 帧
  --skip_frames_start 0 \  # 跳过视频开头的帧数为 0
  --skip_frames_end 0 \  # 跳过视频结尾的帧数为 0
  --train_batch_size 4 \  # 训练时的 batch size 设置为 4
  --num_train_epochs 30 \  # 总训练epoch数为 30
  --checkpointing_steps 1000 \  # 每 1000 步保存一次模型检查点
  --gradient_accumulation_steps 1 \  # 梯度累计步数为 1，即每个 batch 后都会更新梯度
  --learning_rate 1e-3 \  # 学习率设置为 0.001
  --lr_scheduler cosine_with_restarts \  # 使用带重启的余弦学习率调度器
  --lr_warmup_steps 200 \  # 在训练的前 200 步进行学习率预热
  --lr_num_cycles 1 \  # 学习率周期设置为 1
  --optimizer AdamW \  # 使用 AdamW 优化器
  --adam_beta1 0.9 \  # 设置 Adam 优化器的 beta1 参数为 0.9
  --adam_beta2 0.95 \  # 设置 Adam 优化器的 beta2 参数为 0.95
  --max_grad_norm 1.0 \  # 最大梯度裁剪值设置为 1.0
  --allow_tf32 \  # 启用 TF32 以加速训练
  --report_to wandb  # 使用 Weights and Biases 进行训练记录与监控
```

## 运行脚本，开始微调

单机(单卡，多卡)微调：

```shell
bash finetune_single_rank.sh
```

多机多卡微调：

```shell
bash finetune_multi_rank.sh #需要在每个节点运行
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

