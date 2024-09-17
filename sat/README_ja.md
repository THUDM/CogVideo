# SAT CogVideoX-2B

[Read this in English.](./README_zh)

[中文阅读](./README_zh.md)

このフォルダには、[SAT](https://github.com/THUDM/SwissArmyTransformer) ウェイトを使用した推論コードと、SAT
ウェイトのファインチューニングコードが含まれています。

このコードは、チームがモデルをトレーニングするために使用したフレームワークです。コメントが少なく、注意深く研究する必要があります。

## 推論モデル

### 1. このフォルダに必要な依存関係が正しくインストールされていることを確認してください。

```shell
pip install -r requirements.txt
```

### 2. モデルウェイトをダウンロードします

まず、SAT ミラーに移動してモデルの重みをダウンロードします。 CogVideoX-2B モデルの場合は、次のようにダウンロードしてください。

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

CogVideoX-5B モデルの `transformers` ファイルを以下のリンクからダウンロードしてください （VAE ファイルは 2B と同じです）：

+ [CogVideoX-5B](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX-5B-I2V](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)

次に、モデルファイルを以下の形式にフォーマットする必要があります：

```
.
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

モデルの重みファイルが大きいため、`git lfs`を使用することをお勧めいたします。`git lfs`
のインストールについては、[こちら](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing)をご参照ください。

```shell
git lfs install
```

次に、T5 モデルをクローンします。これはトレーニングやファインチューニングには使用されませんが、使用する必要があります。
> モデルを複製する際には、[Modelscope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b)のモデルファイルの場所もご使用いただけます。

```shell
git clone https://huggingface.co/THUDM/CogVideoX-2b.git #ハギングフェイス(huggingface.org)からモデルをダウンロードいただきます
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git #Modelscopeからモデルをダウンロードいただきます
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

### 3. `configs/cogvideox_2b.yaml` ファイルを変更します。

```yaml
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  log_keys:
    - txt

  denoiser_config:
    target: sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser
    params:
      num_idx: 1000
      quantize_c_noise: False

      weighting_config:
        target: sgm.modules.diffusionmodules.denoiser_weighting.EpsWeighting
      scaling_config:
        target: sgm.modules.diffusionmodules.denoiser_scaling.VideoScaling
      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

  network_config:
    target: dit_video_concat.DiffusionTransformer
    params:
      time_embed_dim: 512
      elementwise_affine: True
      num_frames: 49
      time_compressed_rate: 4
      latent_width: 90
      latent_height: 60
      num_layers: 30
      patch_size: 2
      in_channels: 16
      out_channels: 16
      hidden_size: 1920
      adm_in_channels: 256
      num_attention_heads: 30

      transformer_args:
        checkpoint_activations: True ## グラデーション チェックポイントを使用する
        vocab_size: 1
        max_sequence_length: 64
        layernorm_order: pre
        skip_init: false
        model_parallel_size: 1
        is_decoder: false

      modules:
        pos_embed_config:
          target: dit_video_concat.Basic3DPositionEmbeddingMixin
          params:
            text_length: 226
            height_interpolation: 1.875
            width_interpolation: 1.875

        patch_embed_config:
          target: dit_video_concat.ImagePatchEmbeddingMixin
          params:
            text_hidden_size: 4096

        adaln_layer_config:
          target: dit_video_concat.AdaLNMixin
          params:
            qk_ln: True

        final_layer_config:
          target: dit_video_concat.FinalLayerMixin

  conditioner_config:
    target: sgm.modules.GeneralConditioner
    params:
      emb_models:
        - is_trainable: false
          input_key: txt
          ucg_rate: 0.1
          target: sgm.modules.encoders.modules.FrozenT5Embedder
          params:
            model_dir: "t5-v1_1-xxl" # CogVideoX-2b/t5-v1_1-xxlフォルダの絶対パス
            max_length: 226

  first_stage_config:
    target: vae_modules.autoencoder.VideoAutoencoderInferenceWrapper
    params:
      cp_size: 1
      ckpt_path: "CogVideoX-2b-sat/vae/3d-vae.pt" # CogVideoX-2b-sat/vae/3d-vae.ptフォルダの絶対パス
      ignore_keys: [ 'loss' ]

      loss_config:
        target: torch.nn.Identity

      regularizer_config:
        target: vae_modules.regularizers.DiagonalGaussianRegularizer

      encoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelEncoder3D
        params:
          double_z: true
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: True

      decoder_config:
        target: vae_modules.cp_enc_dec.ContextParallelDecoder3D
        params:
          double_z: True
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1, 2, 2, 4 ]
          attn_resolutions: [ ]
          num_res_blocks: 3
          dropout: 0.0
          gather_norm: False

  loss_fn_config:
    target: sgm.modules.diffusionmodules.loss.VideoDiffusionLoss
    params:
      offset_noise_level: 0
      sigma_sampler_config:
        target: sgm.modules.diffusionmodules.sigma_sampling.DiscreteSampling
        params:
          uniform_sampling: True
          num_idx: 1000
          discretization_config:
            target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
            params:
              shift_scale: 3.0

  sampler_config:
    target: sgm.modules.diffusionmodules.sampling.VPSDEDPMPP2MSampler
    params:
      num_steps: 50
      verbose: True

      discretization_config:
        target: sgm.modules.diffusionmodules.discretizer.ZeroSNRDDPMDiscretization
        params:
          shift_scale: 3.0

      guider_config:
        target: sgm.modules.diffusionmodules.guiders.DynamicCFG
        params:
          scale: 6
          exp: 5
          num_steps: 50
```

### 4. `configs/inference.yaml` ファイルを変更します。

```yaml
args:
  latent_channels: 16
  mode: inference
  load: "{absolute_path/to/your}/transformer" # CogVideoX-2b-sat/transformerフォルダの絶対パス
  # load: "{your lora folder} such as zRzRzRzRzRzRzR/lora-disney-08-20-13-28" # This is for Full model without lora adapter

  batch_size: 1
  input_type: txt #TXTのテキストファイルを入力として選択されたり、CLIコマンドラインを入力として変更されたりいただけます
  input_file: configs/test.txt #テキストファイルのパスで、これに対して編集がさせていただけます
  sampling_num_frames: 13  # Must be 13, 11 or 9
  sampling_fps: 8
  fp16: True # For CogVideoX-2B
  #  bf16: True # For CogVideoX-5B
  output_dir: outputs/
  force_inference: True
```

+ 複数のプロンプトを保存するために txt を使用する場合は、`configs/test.txt`
  を参照して変更してください。1行に1つのプロンプトを記述します。プロンプトの書き方がわからない場合は、最初に [このコード](../inference/convert_demo.py)
  を使用して LLM によるリファインメントを呼び出すことができます。
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

### 5. 推論コードを実行して推論を開始します。

```shell
bash inference.sh
```

## モデルのファインチューニング

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

`Lora` とフルパラメータ微調整の2つの方法をサポートしています。両方の微調整方法は、`transformer` 部分のみを微調整し、`VAE`
部分には変更を加えないことに注意してください。`T5` はエンコーダーとしてのみ使用されます。以下のように `configs/sft.yaml` (
フルパラメータ微調整用) ファイルを変更してください。

```
  # checkpoint_activations: True ## 勾配チェックポイントを使用する場合 (設定ファイル内の2つの checkpoint_activations を True に設定する必要があります)
  model_parallel_size: 1 # モデル並列サイズ
  experiment_name: lora-disney  # 実験名 (変更しないでください)
  mode: finetune # モード (変更しないでください)
  load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer モデルのパス
  no_load_rng: True # 乱数シードを読み込むかどうか
  train_iters: 1000 # トレーニングイテレーション数
  eval_iters: 1 # 評価イテレーション数
  eval_interval: 100    # 評価間隔
  eval_batch_size: 1  # 評価バッチサイズ
  save: ckpts # モデル保存パス
  save_interval: 100 # モデル保存間隔
  log_interval: 20 # ログ出力間隔
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # トレーニングデータと評価データは同じでも構いません
  split: 1,0,0 # トレーニングセット、評価セット、テストセットの割合
  num_workers: 8 # データローダーのワーカースレッド数
  force_train: True # チェックポイントをロードするときに欠落したキーを許可 (T5 と VAE は別々にロードされます)
  only_log_video_latents: True # VAE のデコードによるメモリオーバーヘッドを回避
  deepspeed:
    bf16:
      enabled: False # CogVideoX-2B の場合は False に設定し、CogVideoX-5B の場合は True に設定
    fp16:
      enabled: True  # CogVideoX-2B の場合は True に設定し、CogVideoX-5B の場合は False に設定
```

Lora 微調整を使用したい場合は、`cogvideox_<model_parameters>_lora` ファイルも変更する必要があります。

ここでは、`CogVideoX-2B` を参考にします。

```
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## コメントを解除
  log_keys:
    - txt'

  lora_config: ## コメントを解除
    target: sat.model.finetune.lora2.LoraMixin
    params:
      r: 256
```

### 実行スクリプトの変更

設定ファイルを選択するために `finetune_single_gpu.sh` または `finetune_multi_gpus.sh` を編集します。以下に2つの例を示します。

1. `CogVideoX-2B` モデルを使用し、`Lora` 手法を利用する場合は、`finetune_single_gpu.sh` または `finetune_multi_gpus.sh`
   を変更する必要があります。

```
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
```

2. `CogVideoX-2B` モデルを使用し、`フルパラメータ微調整` 手法を利用する場合は、`finetune_single_gpu.sh`
   または `finetune_multi_gpus.sh` を変更する必要があります。

```
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b.yaml configs/sft.yaml --seed $RANDOM"
```

### 微調整と評価

推論コードを実行して微調整を開始します。

```
bash finetune_single_gpu.sh # シングルGPU
bash finetune_multi_gpus.sh # マルチGPU
```

### 微調整後のモデルの使用

微調整されたモデルは統合できません。ここでは、推論設定ファイル `inference.sh` を変更する方法を示します。

```
run_cmd="$environs python sample_video.py --base configs/cogvideox_<model_parameters>_lora.yaml configs/inference.yaml --seed 42"
```

その後、次のコードを実行します。

```
bash inference.sh 
```

### Huggingface Diffusers サポートのウェイトに変換

SAT ウェイト形式は Huggingface のウェイト形式と異なり、変換が必要です。次のコマンドを実行してください：

```shell
python ../tools/convert_weight_sat2hf.py
```

### SATチェックポイントからHuggingface Diffusers lora LoRAウェイトをエクスポート

上記のステップを完了すると、LoRAウェイト付きのSATチェックポイントが得られます。ファイルは `{args.save}/1000/1000/mp_rank_00_model_states.pt` にあります。

LoRAウェイトをエクスポートするためのスクリプトは、CogVideoXリポジトリの `tools/export_sat_lora_weight.py` にあります。エクスポート後、`load_cogvideox_lora.py` を使用して推論を行うことができます。

エクスポートコマンド:

```bash
python tools/export_sat_lora_weight.py --sat_pt_path {args.save}/{experiment_name}-09-09-21-10/1000/mp_rank_00_model_states.pt --lora_save_directory {args.save}/export_hf_lora_weights_1/
```

このトレーニングでは主に以下のモデル構造が変更されました。以下の表は、HF (Hugging Face) 形式のLoRA構造に変換する際の対応関係を示しています。ご覧の通り、LoRAはモデルの注意メカニズムに低ランクの重みを追加しています。

```
'attention.query_key_value.matrix_A.0': 'attn1.to_q.lora_A.weight',
'attention.query_key_value.matrix_A.1': 'attn1.to_k.lora_A.weight',
'attention.query_key_value.matrix_A.2': 'attn1.to_v.lora_A.weight',
'attention.query_key_value.matrix_B.0': 'attn1.to_q.lora_B.weight',
'attention.query_key_value.matrix_B.1': 'attn1.to_k.lora_B.weight',
'attention.query_key_value.matrix_B.2': 'attn1.to_v.lora_B.weight',
'attention.dense.matrix_A.0': 'attn1.to_out.0.lora_A.weight',
'attention.dense.matrix_B.0': 'attn1.to_out.0.lora_B.weight'
```
  
export_sat_lora_weight.py を使用して、SATチェックポイントをHF LoRA形式に変換できます。


![alt text](../resources/hf_lora_weights.png)
