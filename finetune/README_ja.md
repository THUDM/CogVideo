# CogVideoX diffusers 微調整方法

[Read this in English.](./README_zh)

[中文阅读](./README_zh.md)


この機能はまだ完全に完成していません。SATバージョンの微調整を確認したい場合は、[こちら](../sat/README_ja.md)を参照してください。本バージョンとは異なるデータセット形式を使用しています。

## ハードウェア要件

+ CogVideoX-2B LORA: 1 * A100
+ CogVideoX-2B SFT:  8 * A100
+ CogVideoX-5B/5B-I2V まだサポートしていません

## 依存関係のインストール

関連コードはまだdiffusersのリリース版に統合されていないため、diffusersブランチを使用して微調整を行う必要があります。以下の手順に従って依存関係をインストールしてください：

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout cogvideox-lora-and-training
pip install -e .
```

## データセットの準備

まず、データセットを準備する必要があります。データセットの形式は以下のようになります。

```
.
├── prompts.txt
├── videos
└── videos.txt
```

[ディズニースチームボートウィリー](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset)をここからダウンロードできます。

ビデオ微調整データセットはテスト用として使用されます。

## 設定ファイルと実行

`accelerate` 設定ファイルは以下の通りです:

+ accelerate_config_machine_multi.yaml 複数GPU向け
+ accelerate_config_machine_single.yaml 単一GPU向け

`finetune` スクリプト設定ファイルの例：

```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  
# このコマンドは、OOM（メモリ不足）エラーを防ぐために、CUDAメモリ割り当てを拡張セグメントに設定します。

accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu # 複数のGPUで `accelerate` を使用してトレーニングを開始します。指定された設定ファイルを使用します。

  train_cogvideox_lora.py   # LoRA微調整用に CogVideoX モデルをトレーニングするスクリプトです。

  --pretrained_model_name_or_path THUDM/CogVideoX-2b   # 事前学習済みモデルのパスです。

  --cache_dir ~/.cache   # Hugging Faceからダウンロードされたモデルとデータセットのキャッシュディレクトリです。

  --enable_tiling   # VAEタイル化機能を有効にし、メモリ使用量を削減します。

  --enable_slicing   # VAEスライス機能を有効にして、チャネルでのスライス処理を行い、メモリを節約します。

  --instance_data_root ~/disney/   # インスタンスデータのルートディレクトリです。

  --caption_column prompts.txt   # テキストプロンプトが含まれているファイルや列を指定します。

  --video_column videos.txt   # ビデオパスが含まれているファイルや列を指定します。

  --validation_prompt "Mickey with the captain and friends:::Mickey and the bear"   # 検証用のプロンプトを指定します。複数のプロンプトを指定するには `:::` 区切り文字を使用します。

  --validation_prompt_separator :::   # 検証プロンプトの区切り文字を `:::` に設定します。

  --num_validation_videos 1   # 検証中に生成するビデオの数を1に設定します。

  --validation_epochs 2   # 何エポックごとに検証を行うかを2に設定します。

  --seed 3407   # ランダムシードを3407に設定し、トレーニングの再現性を確保します。

  --rank 128   # LoRAの更新マトリックスの次元を128に設定します。

  --mixed_precision bf16   # 混合精度トレーニングを `bf16` (bfloat16) に設定します。

  --output_dir cogvideox-lora-single-gpu   # 出力ディレクトリを指定します。

  --height 480   # 入力ビデオの高さを480ピクセルに設定します。

  --width 720   # 入力ビデオの幅を720ピクセルに設定します。

  --fps 8   # 入力ビデオのフレームレートを8 fpsに設定します。

  --max_num_frames 49   # 入力ビデオの最大フレーム数を49に設定します。

  --skip_frames_start 0   # 各ビデオの最初のフレームをスキップしません。

  --skip_frames_end 0   # 各ビデオの最後のフレームをスキップしません。

  --train_batch_size 1   # トレーニングバッチサイズを1に設定します。

  --num_train_epochs 10   # トレーニングのエポック数を10に設定します。

  --checkpointing_steps 500   # 500ステップごとにチェックポイントを保存します。

  --gradient_accumulation_steps 1   # 1ステップごとに勾配を蓄積して更新します。

  --learning_rate 1e-4   # 初期学習率を1e-4に設定します。

  --optimizer AdamW   # AdamWオプティマイザーを使用します。

  --adam_beta1 0.9   # Adamのbeta1パラメータを0.9に設定します。

  --adam_beta2 0.95   # Adamのbeta2パラメータを0.95に設定します。
```

## 微調整を開始

単一GPU微調整：

```shell
bash finetune_single_gpu.sh
```

複数GPU微調整：

```shell
bash finetune_multi_gpus_1.sh # 各ノードで実行する必要があります。
```

## 微調整済みモデルのロード

+ 微調整済みのモデルをロードする方法については、[cli_demo.py](../inference/cli_demo.py) を参照してください。

## ベストプラクティス

+ 解像度が `200 x 480 x 720`（フレーム数 x 高さ x 幅）のトレーニングビデオが70本含まれています。データ前処理でフレームをスキップすることで、49フレームと16フレームの小さなデータセットを作成しました。これは実験を加速するためのもので、CogVideoXチームが推奨する最大フレーム数制限は49フレームです。
+ 25本以上のビデオが新しい概念やスタイルのトレーニングに最適です。
+ 現在、`--id_token` を指定して識別トークンを使用してトレーニングする方が効果的です。これはDreamboothトレーニングに似ていますが、通常の微調整でも機能します。
+ 元のリポジトリでは `lora_alpha` を1に設定していましたが、複数の実行でこの値が効果的でないことがわかりました。モデルのバックエンドやトレーニング設定によるかもしれません。私たちの提案は、lora_alphaをrankと同じか、rank // 2に設定することです。
+ Rank 64以上の設定を推奨します。
