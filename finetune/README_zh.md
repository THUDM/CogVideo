# CogVideoX diffusers 微调方案

[Read this in English](./README.md)

[日本語で読む](./README_ja.md)

如果您想查看SAT版本微调，请查看[这里](../sat/README_zh.md)。其数据集格式与本版本不同。

## 安装依赖

由于相关代码还没有被合并到diffusers发行版，你需要基于diffusers分支进行微调。请按照以下步骤安装依赖：

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now in Main branch
pip install -e .
```

## 准备数据集

首先，你需要准备数据集。根据你的任务类型（T2V 或 I2V），数据集格式略有不同：

```
.
├── prompts.txt
├── videos
├── videos.txt
├── images # 仅 I2V 需要
└── images.txt # 仅 I2V 需要
```

其中：
- `prompts.txt`: 存放提示词
- `videos/`: 存放.mp4视频文件
- `videos.txt`: 存放 videos 目录中的视频文件列表
- `images/`: 存放.png参考图像文件
- `images.txt`: 存放参考图像文件列表

你可以从这里下载示例数据集(T2V) [迪士尼汽船威利号](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset)

> 我们在[这里](./scripts/extract_images.py)提供了提取视频第一帧为图片的脚本，对于I2V任务您可以使用它来生成参考图像。

如果需要在训练过程中进行validation，则需要额外提供验证数据集，其中数据格式与训练集相同。

## 运行脚本，开始微调

在开始训练之前，请注意以下分辨率设置要求：

1. 帧数必须是8的倍数 **+1** (即8N+1), 例如49, 81 ...
2. 视频分辨率建议使用模型的默认大小：
   - CogVideoX: 480x720 (高x宽)
   - CogVideoX1.5: 768x1360 (高x宽)
3. 对于不满足训练分辨率的样本（视频或图片）在代码中会直接进行resize。这可能会导致样本的宽高比发生形变从而影响训练效果。建议用户提前对样本在分辨率上进行处理（例如使用crop + resize来维持宽高比）再进行训练。

> **重要提示**：为了提高训练效率，我们会在训练前自动对video进行encode并将结果缓存在磁盘。如果在训练后修改了数据，请删除video目录下的latent目录，以确保使用最新的数据。

### 文本生成视频 (T2V) 微调

```bash
# 修改 accelerate_train_t2v.sh 中的配置参数
# 主要需要修改以下参数:
# --output_dir: 输出目录
# --data_root: 数据集根目录
# --caption_column: 提示词文件路径
# --video_column: 视频文件列表路径
# --train_resolution: 训练分辨率 (帧数x高x宽)
# 其他重要参数请参考启动脚本

bash accelerate_train_t2v.sh
```

### 图像生成视频 (I2V) 微调

```bash
# 修改 accelerate_train_i2v.sh 中的配置参数
# 除了需要修改与T2V相同的参数外，还需要额外设置:
# --image_column: 参考图像文件列表路径
# 其他重要参数请参考启动脚本

bash accelerate_train_i2v.sh
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

