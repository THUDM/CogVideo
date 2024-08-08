# SAT CogVideoX-2B

このフォルダには、[SAT](https://github.com/THUDM/SwissArmyTransformer) ウェイトを使用した推論コードと、SAT ウェイトのファインチューニングコードが含まれています。

このコードは、チームがモデルをトレーニングするために使用したフレームワークです。コメントが少なく、注意深く研究する必要があります。

## 推論モデル

1. このフォルダに必要な依存関係が正しくインストールされていることを確認してください。

```shell
pip install -r requirements.txt
```

2. モデルウェイトをダウンロードします

まず、SAT ミラーにアクセスして依存関係をダウンロードします。

```shell
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
```

次に解凍し、モデル構造は次のようになります：

```
.
├── transformer
│   ├── 1000
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

次に、T5 モデルをクローンします。これはトレーニングやファインチューニングには使用されませんが、使用する必要があります。

```
git clone https://huggingface.co/THUDM/CogVideoX-2b.git
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

上記の方法に従うことで、safetensor 形式の T5 ファイルを取得できます。これにより、Deepspeed でのファインチューニング中にエラーが発生しないようにします。

```
├── added_tokens.json
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json

0 directories, 8 files
```

3. `configs/cogvideox_2b_infer.yaml` ファイルを変更します。

```yaml
load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer モデルパス

conditioner_config:
  target: sgm.modules.GeneralConditioner
  params:
    emb_models:
      - is_trainable: false
        input_key: txt
        ucg_rate: 0.1
        target: sgm.modules.encoders.modules.FrozenT5Embedder
        params:
          model_dir: "google/t5-v1_1-xxl" ## T5 モデルパス
          max_length: 226

first_stage_config:
  target: sgm.models.autoencoder.VideoAutoencoderInferenceWrapper
  params:
    cp_size: 1
    ckpt_path: "{your_CogVideoX-2b-sat_path}/vae/3d-vae.pt" ## VAE モデルパス
```

+ 複数のプロンプトを保存するために txt を使用する場合は、`configs/test.txt` を参照して変更してください。1行に1つのプロンプトを記述します。プロンプトの書き方がわからない場合は、最初に [このコード](../inference/convert_demo.py) を使用して LLM によるリファインメントを呼び出すことができます。
+ コマンドラインを入力として使用する場合は、次のように変更します。

```yaml
input_type: cli
```

これにより、コマンドラインからプロンプトを入力できます。

出力ビデオのディレクトリを変更したい場合は、次のように変更できます：

```yaml
output_dir: outputs/
```

デフォルトでは `.outputs/` フォルダに保存されます。

4. 推論コードを実行して推論を開始します。

```shell
bash inference.sh
```

## モデルのファインチューニング

### 環境の準備

現在、SAT はソースコードからインストールする必要があり、正常にファインチューニングを行うためにはこれが必要です。この問題は将来の安定版で解決される予定です。

```
git clone https://github.com/THUDM/SwissArmyTransformer.git
cd SwissArmyTransformer
pip install -e .
```

### データセットの準備

データセットの形式は次のようになります：

```
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```

各 txt ファイルは対応するビデオファイルと同じ名前であり、そのビデオのラベルを含んでいます。各ビデオはラベルと一対一で対応する必要があります。通常、1つのビデオに複数のラベルを持たせることはありません。

スタイルファインチューニングの場合、少なくとも50本のスタイルが似たビデオとラベルを準備し、フィッティングを容易にします。

### 設定ファイルの変更

`Lora` と 全パラメータファインチューニングの2つの方法をサポートしています。これらのファインチューニング方法は `transformer` 部分にのみ適用されます。`VAE` 部分は変更されません。`T5` はエンコーダーとしてのみ使用されます。

`configs/cogvideox_2b_sft.yaml` (全量ファインチューニング用) を次のように変更します。

```yaml
  # checkpoint_activations: True ## using gradient checkpointing (設定ファイル内の2つのcheckpoint_activationsを両方Trueに設定する必要があります)
  model_parallel_size: 1 # モデル並列サイズ
  experiment_name: lora-disney  # 実験名 (変更しないでください)
  mode: finetune # モード (変更しないでください)
  load: "{your_CogVideoX-2b-sat_path}/transformer" # Transformer モデルパス
  no_load_rng: True # ランダムシードをロードするかどうか
  train_iters: 1000 # トレーニングイテレーション数
  eval_iters: 1 # 評価イテレーション数
  eval_interval: 100 # 評価間隔
  eval_batch_size: 1 # 評価のバッチサイズ
  save: ckpts # モデル保存パス
  save_interval: 100 # モデル保存間隔
  log_interval: 20 # ログ出力間隔
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # トレーニングセットと検証セットは同じでもかまいません
  split: 1,0,0 # トレーニングセット、検証セット、テストセットの比率
  num_workers: 8 # データローダーのワーカースレッド数
  force_train: True # ckpt をロードする際に missing keys を許可するかどうか (T5 と VAE は独立してロードされます)
  only_log_video_latents: True # VAE デコーダーを使用しないようにしてメモリを節約します
```

Lora ファインチューニングを使用する場合は、次のように変更する必要があります：

```yaml
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## コメント解除
  log_keys:
    - txt'

  lora_config: ## コメント解除
    target: sat.model.finetune.lora2.LoraMixin
    params:
      r: 256
```

### ファインチューニングと検証

1. 推論コードを実行してファインチューニングを開始します。

```shell
bash finetune.sh
```

### Huggingface Diffusers サポートのウェイトに変換

SAT ウェイト形式は Huggingface のウェイト形式と異なり、変換が必要です。次のコマンドを実行してください：

```shell
python ../tools/convert_weight_sat2hf.py
```

**注意**：この内容は LORA ファインチューニングモデルではまだテストされていません。
