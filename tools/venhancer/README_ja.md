
# VEnhancer で CogVideoX によって生成されたビデオを強化する

このチュートリアルでは、VEnhancer ツールを使用して、CogVideoX で生成されたビデオを強化し、より高いフレームレートと高い解像度を実現する方法を説明します。

## モデルの紹介

VEnhancer は、空間超解像、時間超解像（フレーム補間）、およびビデオのリファインメントを統一されたフレームワークで実現します。空間または時間の超解像のために、さまざまなアップサンプリング係数（例：1x〜8x）に柔軟に対応できます。さらに、多様なビデオアーティファクトを処理するために、リファインメント強度を変更する柔軟な制御を提供します。

VEnhancer は ControlNet の設計に従い、事前訓練されたビデオ拡散モデルのマルチフレームエンコーダーとミドルブロックのアーキテクチャとウェイトをコピーして、トレーニング可能な条件ネットワークを構築します。このビデオ ControlNet は、低解像度のキーフレームとノイズを含む完全なフレームを入力として受け取ります。さらに、タイムステップ t とプロンプトに加えて、提案されたビデオ対応条件により、ノイズ増幅レベル σ およびダウンスケーリングファクター s が追加のネットワーク条件として使用されます。

## ハードウェア要件

+ オペレーティングシステム: Linux (xformers 依存関係が必要)
+ ハードウェア: 単一カードあたり少なくとも 60GB の VRAM を持つ NVIDIA GPU。H100、A100 などのマシンを推奨します。

## クイックスタート

1. 公式の指示に従ってリポジトリをクローンし、依存関係をインストールします。

```shell
git clone https://github.com/Vchitect/VEnhancer.git
cd VEnhancer
## Torch などの依存関係は CogVideoX の依存関係を使用できます。新しい環境を作成する必要がある場合は、以下のコマンドを使用してください。
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

## 必須の依存関係をインストールします。
pip install -r requirements.txt
```

2. コードを実行します。

```shell
python enhance_a_video.py --up_scale 4 --target_fps 24 --noise_aug 250 --solver_mode 'fast' --steps 15 --input_path inputs/000000.mp4 --prompt 'Wide-angle aerial shot at dawn, soft morning light casting long shadows, an elderly man walking his dog through a quiet, foggy park, trees and benches in the background, peaceful and serene atmosphere' --save_dir 'results/'
```

次の設定を行います：

- `input_path` 是输入视频的路径
- `prompt` 是视频内容的描述。此工具使用的提示词应更短，不超过77个字。您可能需要简化用于生成CogVideoX视频的提示词。
- `target_fps` 是视频的目标帧率。通常，16 fps已经很流畅，默认值为24 fps。
- `up_scale` 推荐设置为2、3或4。目标分辨率限制在2k左右及以下。
- `noise_aug` 的值取决于输入视频的质量。质量较低的视频需要更高的噪声级别，这对应于更强的优化。250~300适用于非常低质量的视频。对于高质量视频，设置为≤200。
- `steps` 如果想减少步数，请先将solver_mode改为“normal”，然后减少步数。“fast”模式的步数是固定的（15步）。
  代码在执行过程中会自动从Hugging Face下载所需的模型。

コードの実行中に、必要なモデルは Hugging Face から自動的にダウンロードされます。

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

A100 GPU を単一で使用している場合、CogVideoX によって生成された 6 秒間のビデオを強化するには、デフォルト設定で 60GB の VRAM を消費し、40〜50 分かかります。
