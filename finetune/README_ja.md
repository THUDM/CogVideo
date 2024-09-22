# CogVideoX diffusers 微調整方法

[Read this in English.](./README_zh)

[中文阅读](./README_zh.md)


この機能はまだ完全に完成していません。SATバージョンの微調整を確認したい場合は、[こちら](../sat/README_ja.md)を参照してください。本バージョンとは異なるデータセット形式を使用しています。

## ハードウェア要件

+ CogVideoX-2B / 5B T2V LORA: 1 * A100  (5B need to use `--use_8bit_adam`)
+ CogVideoX-2B SFT:  8 * A100 （動作確認済み）
+ CogVideoX-5B-I2V まだサポートしていません

## 依存関係のインストール

関連コードはまだdiffusersのリリース版に統合されていないため、diffusersブランチを使用して微調整を行う必要があります。以下の手順に従って依存関係をインストールしてください：

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now in Main branch
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

```
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \  # accelerateを使用してmulti-GPUトレーニングを起動、設定ファイルはaccelerate_config_machine_single.yaml
  train_cogvideox_lora.py \  # LoRAの微調整用のトレーニングスクリプトtrain_cogvideox_lora.pyを実行
  --gradient_checkpointing \  # メモリ使用量を減らすためにgradient checkpointingを有効化
  --pretrained_model_name_or_path $MODEL_PATH \  # 事前学習済みモデルのパスを$MODEL_PATHで指定
  --cache_dir $CACHE_PATH \  # モデルファイルのキャッシュディレクトリを$CACHE_PATHで指定
  --enable_tiling \  # メモリ節約のためにタイル処理を有効化し、動画をチャンク分けして処理
  --enable_slicing \  # 入力をスライスしてさらにメモリ最適化
  --instance_data_root $DATASET_PATH \  # データセットのパスを$DATASET_PATHで指定
  --caption_column prompts.txt \  # トレーニングで使用する動画の説明ファイルをprompts.txtで指定
  --video_column videos.txt \  # トレーニングで使用する動画のパスファイルをvideos.txtで指定
  --validation_prompt "" \  # トレーニング中に検証用の動画を生成する際のプロンプト
  --validation_prompt_separator ::: \  # 検証プロンプトの区切り文字を:::に設定
  --num_validation_videos 1 \  # 各検証ラウンドで1本の動画を生成
  --validation_epochs 100 \  # 100エポックごとに検証を実施
  --seed 42 \  # 再現性を保証するためにランダムシードを42に設定
  --rank 128 \  # LoRAのパラメータのランクを128に設定
  --lora_alpha 64 \  # LoRAのalphaパラメータを64に設定し、LoRAの学習率を調整
  --mixed_precision bf16 \  # bf16混合精度でトレーニングし、メモリを節約
  --output_dir $OUTPUT_PATH \  # モデルの出力ディレクトリを$OUTPUT_PATHで指定
  --height 480 \  # 動画の高さを480ピクセルに設定
  --width 720 \  # 動画の幅を720ピクセルに設定
  --fps 8 \  # 動画のフレームレートを1秒あたり8フレームに設定
  --max_num_frames 49 \  # 各動画の最大フレーム数を49に設定
  --skip_frames_start 0 \  # 動画の最初のフレームを0スキップ
  --skip_frames_end 0 \  # 動画の最後のフレームを0スキップ
  --train_batch_size 4 \  # トレーニングのバッチサイズを4に設定
  --num_train_epochs 30 \  # 総トレーニングエポック数を30に設定
  --checkpointing_steps 1000 \  # 1000ステップごとにモデルのチェックポイントを保存
  --gradient_accumulation_steps 1 \  # 1ステップの勾配累積を行い、各バッチ後に更新
  --learning_rate 1e-3 \  # 学習率を0.001に設定
  --lr_scheduler cosine_with_restarts \  # リスタート付きのコサイン学習率スケジューラを使用
  --lr_warmup_steps 200 \  # トレーニングの最初の200ステップで学習率をウォームアップ
  --lr_num_cycles 1 \  # 学習率のサイクル数を1に設定
  --optimizer AdamW \  # AdamWオプティマイザーを使用
  --adam_beta1 0.9 \  # Adamオプティマイザーのbeta1パラメータを0.9に設定
  --adam_beta2 0.95 \  # Adamオプティマイザーのbeta2パラメータを0.95に設定
  --max_grad_norm 1.0 \  # 勾配クリッピングの最大値を1.0に設定
  --allow_tf32 \  # トレーニングを高速化するためにTF32を有効化
  --report_to wandb  # Weights and Biasesを使用してトレーニングの記録とモニタリングを行う
```

## 微調整を開始

単一マシン (シングルGPU、マルチGPU) での微調整:

```shell
bash finetune_single_rank.sh
```

複数マシン・マルチGPUでの微調整：

```shell
bash finetune_multi_rank.sh # 各ノードで実行する必要があります。
```

## 微調整済みモデルのロード

+ 微調整済みのモデルをロードする方法については、[cli_demo.py](../inference/cli_demo.py) を参照してください。

## ベストプラクティス

+ 解像度が `200 x 480 x 720`（フレーム数 x 高さ x 幅）のトレーニングビデオが70本含まれています。データ前処理でフレームをスキップすることで、49フレームと16フレームの小さなデータセットを作成しました。これは実験を加速するためのもので、CogVideoXチームが推奨する最大フレーム数制限は49フレームです。
+ 25本以上のビデオが新しい概念やスタイルのトレーニングに最適です。
+ 現在、`--id_token` を指定して識別トークンを使用してトレーニングする方が効果的です。これはDreamboothトレーニングに似ていますが、通常の微調整でも機能します。
+ 元のリポジトリでは `lora_alpha` を1に設定していましたが、複数の実行でこの値が効果的でないことがわかりました。モデルのバックエンドやトレーニング設定によるかもしれません。私たちの提案は、lora_alphaをrankと同じか、rank // 2に設定することです。
+ Rank 64以上の設定を推奨します。
