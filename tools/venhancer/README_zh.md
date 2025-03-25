# 使用 VEnhancer 对 CogVdieoX 生成视频进行增强

本教程将要使用 VEnhancer 工具 对 CogVdieoX 生成视频进行增强, 包括更高的帧率和更高的分辨率

## 模型介绍

VEnhancer 在一个统一的框架中实现了空间超分辨率、时间超分辨率（帧插值）和视频优化。它可以灵活地适应不同的上采样因子（例如，1x~
8x）用于空间或时间超分辨率。此外，它提供了灵活的控制，以修改优化强度，从而处理多样化的视频伪影。

VEnhancer 遵循 ControlNet 的设计，复制了预训练的视频扩散模型的多帧编码器和中间块的架构和权重，构建了一个可训练的条件网络。这个视频
ControlNet 接受低分辨率关键帧和包含噪声的完整帧作为输入。此外，除了时间步 t 和提示词外，我们提出的视频感知条件还将噪声增强的噪声级别
σ 和降尺度因子 s 作为附加的网络条件输入。

## 硬件需求

+ 操作系统: Linux (需要依赖xformers)
+ 硬件: NVIDIA GPU 并至少保证单卡显存超过60G，推荐使用 H100，A100等机器。

## 快速上手

1. 按照官方指引克隆仓库并安装依赖

```shell
git clone https://github.com/Vchitect/VEnhancer.git
cd VEnhancer
## torch等依赖可以使用CogVideoX的依赖，如果你需要创建一个新的环境，可以使用以下命令
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

## 安装必须的依赖
pip install -r requirements.txt
```

2. 运行代码

```shell
python enhance_a_video.py \
--up_scale 4 --target_fps 24 --noise_aug 250 \
--solver_mode 'fast' --steps 15 \
--input_path inputs/000000.mp4 \
--prompt 'Wide-angle aerial shot at dawn,soft morning light casting long shadows,an elderly man walking his dog through a quiet,foggy park,trees and benches in the background,peaceful and serene atmosphere' \
--save_dir 'results/'
```

其中:

- `input_path` 是输入视频的路径
- `prompt` 是视频内容的描述。此工具使用的提示词应更短，不超过77个字。您可能需要简化用于生成CogVideoX视频的提示词。
- `target_fps` 是视频的目标帧率。通常，16 fps已经很流畅，默认值为24 fps。
- `up_scale` 推荐设置为2、3或4。目标分辨率限制在2k左右及以下。
- `noise_aug` 的值取决于输入视频的质量。质量较低的视频需要更高的噪声级别，这对应于更强的优化。250~300适用于非常低质量的视频。对于高质量视频，设置为≤200。
- `steps` 如果想减少步数，请先将solver_mode改为“normal”，然后减少步数。“fast”模式的步数是固定的（15步）。
  代码在执行过程中会自动从Hugging Face下载所需的模型。

代码运行过程中，会自动从Huggingface拉取需要的模型

运行日志通常如下:

```shell
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
2024-08-20 13:25:17,553 - video_to_video - INFO - checkpoint_path: ./ckpts/venhancer_paper.pt
/share/home/zyx/.conda/envs/cogvideox/lib/python3.10/site-packages/open_clip/factory.py:88: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=map_location)
2024-08-20 13:25:37,486 - video_to_video - INFO - Build encoder with FrozenOpenCLIPEmbedder
/share/home/zyx/Code/VEnhancer/video_to_video/video_to_video_model.py:35: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  load_dict = torch.load(cfg.model_path, map_location='cpu')
2024-08-20 13:25:55,391 - video_to_video - INFO - Load model path ./ckpts/venhancer_paper.pt, with local status <All keys matched successfully>
2024-08-20 13:25:55,392 - video_to_video - INFO - Build diffusion with GaussianDiffusion
2024-08-20 13:26:16,092 - video_to_video - INFO - input video path: inputs/000000.mp4
2024-08-20 13:26:16,093 - video_to_video - INFO - text: Wide-angle aerial shot at dawn,soft morning light casting long shadows,an elderly man walking his dog through a quiet,foggy park,trees and benches in the background,peaceful and serene atmosphere
2024-08-20 13:26:16,156 - video_to_video - INFO - input frames length: 49
2024-08-20 13:26:16,156 - video_to_video - INFO - input fps: 8.0
2024-08-20 13:26:16,156 - video_to_video - INFO - target_fps: 24.0
2024-08-20 13:26:16,311 - video_to_video - INFO - input resolution: (480, 720)
2024-08-20 13:26:16,312 - video_to_video - INFO - target resolution: (1320, 1982)
2024-08-20 13:26:16,312 - video_to_video - INFO - noise augmentation: 250
2024-08-20 13:26:16,312 - video_to_video - INFO - scale s is set to: 8
2024-08-20 13:26:16,399 - video_to_video - INFO - video_data shape: torch.Size([145, 3, 1320, 1982])
/share/home/zyx/Code/VEnhancer/video_to_video/video_to_video_model.py:108: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with amp.autocast(enabled=True):
2024-08-20 13:27:19,605 - video_to_video - INFO - step: 0
2024-08-20 13:30:12,020 - video_to_video - INFO - step: 1
2024-08-20 13:33:04,956 - video_to_video - INFO - step: 2
2024-08-20 13:35:58,691 - video_to_video - INFO - step: 3
2024-08-20 13:38:51,254 - video_to_video - INFO - step: 4
2024-08-20 13:41:44,150 - video_to_video - INFO - step: 5
2024-08-20 13:44:37,017 - video_to_video - INFO - step: 6
2024-08-20 13:47:30,037 - video_to_video - INFO - step: 7
2024-08-20 13:50:22,838 - video_to_video - INFO - step: 8
2024-08-20 13:53:15,844 - video_to_video - INFO - step: 9
2024-08-20 13:56:08,657 - video_to_video - INFO - step: 10
2024-08-20 13:59:01,648 - video_to_video - INFO - step: 11
2024-08-20 14:01:54,541 - video_to_video - INFO - step: 12
2024-08-20 14:04:47,488 - video_to_video - INFO - step: 13
2024-08-20 14:10:13,637 - video_to_video - INFO - sampling, finished.

```

使用A100单卡运行，对于每个CogVideoX生产的6秒视频，按照默认配置，会消耗60G显存，并用时40-50分钟。
