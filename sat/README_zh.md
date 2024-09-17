# SAT CogVideoX-2B

[Read this in English.](./README_zh)

[日本語で読む](./README_ja.md)

本文件夹包含了使用 [SAT](https://github.com/THUDM/SwissArmyTransformer) 权重的推理代码，以及 SAT 权重的微调代码。

该代码是团队训练模型时使用的框架。注释较少，需要认真研究。

## 推理模型

### 1. 确保你已经正确安装本文件夹中的要求的依赖

```shell
pip install -r requirements.txt
```

### 2. 下载模型权重

首先，前往 SAT 镜像下载模型权重。

对于 CogVideoX-2B 模型，请按照如下方式下载:

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

请按如下链接方式下载 CogVideoX-5B 模型的 `transformers` 文件（VAE 文件与 2B 相同）：

+ [CogVideoX-5B](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX-5B-I2V](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)

接着，你需要将模型文件排版成如下格式：

```
.
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

由于模型的权重档案较大，建议使用`git lfs`。`git lfs`
安装参见[这里](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing)

```shell
git lfs install
```

接着，克隆 T5 模型，该模型不用做训练和微调，但是必须使用。
> 克隆模型的时候也可以使用[Modelscope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b)上的模型文件位置。

```shell
git clone https://huggingface.co/THUDM/CogVideoX-2b.git #从huggingface下载模型
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git #从modelscope下载模型
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

通过上述方案，你将会得到一个 safetensor 格式的T5文件，确保在 Deepspeed微调过程中读入的时候不会报错。

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

### 3. 修改`configs/cogvideox_2b.yaml`中的文件。

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
        checkpoint_activations: True ## using gradient checkpointing
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
            model_dir: "t5-v1_1-xxl" # CogVideoX-2b/t5-v1_1-xxl 权重文件夹的绝对路径
            max_length: 226

  first_stage_config:
    target: vae_modules.autoencoder.VideoAutoencoderInferenceWrapper
    params:
      cp_size: 1
      ckpt_path: "CogVideoX-2b-sat/vae/3d-vae.pt" # CogVideoX-2b-sat/vae/3d-vae.pt文件夹的绝对路径
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

### 4. 修改`configs/inference.yaml`中的文件。

```yaml
args:
  latent_channels: 16
  mode: inference
  load: "{absolute_path/to/your}/transformer" # CogVideoX-2b-sat/transformer文件夹的绝对路径
  # load: "{your lora folder} such as zRzRzRzRzRzRzR/lora-disney-08-20-13-28" # This is for Full model without lora adapter

  batch_size: 1
  input_type: txt #可以选择txt纯文字档作为输入，或者改成cli命令行作为输入
  input_file: configs/test.txt #纯文字档，可以对此做编辑
  sampling_num_frames: 13  # Must be 13, 11 or 9
  sampling_fps: 8
  fp16: True # For CogVideoX-2B
  #  bf16: True # For CogVideoX-5B
  output_dir: outputs/
  force_inference: True
```

+ 如果使用 txt 保存多个提示词，请参考`configs/test.txt`
  进行修改。每一行一个提示词。如果您不知道如何书写提示词，可以先使用[此代码](../inference/convert_demo.py)调用 LLM进行润色。
+ 如果使用命令行作为输入，请修改

```yaml
input_type: cli
```

这样就可以从命令行输入提示词。

如果你希望修改输出视频的地址，你可以修改:

```yaml
output_dir: outputs/
```

默认保存在`.outputs/`文件夹下。

### 5. 运行推理代码, 即可推理

```shell
bash inference.sh
```

## 微调模型

### 准备数据集

数据集格式应该如下：

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

每个 txt 与视频同名，为视频的标签。视频与标签应该一一对应。通常情况下，不使用一个视频对应多个标签。

如果为风格微调，清准备至少50条风格相似的视频和标签，以利于拟合。

### 修改配置文件

我们支持 `Lora` 和 全参数微调两种方式。请注意，两种微调方式都仅仅对 `transformer` 部分进行微调。不改动 `VAE` 部分。`T5`仅作为
Encoder 使用。
部分。 请按照以下方式修改`configs/sft.yaml`(全量微调) 中的文件。

```yaml
  # checkpoint_activations: True ## using gradient checkpointing (配置文件中的两个checkpoint_activations都需要设置为True)
  model_parallel_size: 1 # 模型并行大小
  experiment_name: lora-disney  # 实验名称(不要改动)
  mode: finetune # 模式(不要改动)
  load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer 模型路径
  no_load_rng: True # 是否加载随机数种子
  train_iters: 1000 # 训练迭代次数
  eval_iters: 1 # 验证迭代次数
  eval_interval: 100    # 验证间隔
  eval_batch_size: 1  # 验证集 batch size
  save: ckpts # 模型保存路径 
  save_interval: 100 # 模型保存间隔
  log_interval: 20 # 日志输出间隔
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # 训练集和验证集可以相同
  split: 1,0,0 # 训练集，验证集，测试集比例
  num_workers: 8 # 数据加载器的工作线程数
  force_train: True # 在加载checkpoint时允许missing keys (T5 和 VAE 单独加载)
  only_log_video_latents: True # 避免VAE decode带来的显存开销
  deepspeed:
    bf16:
      enabled: False # For CogVideoX-2B Turn to False and For CogVideoX-5B Turn to True
    fp16:
      enabled: True  # For CogVideoX-2B Turn to True and For CogVideoX-5B Turn to False
```

如果你希望使用 Lora 微调，你还需要修改`cogvideox_<模型参数>_lora` 文件：

这里以 `CogVideoX-2B` 为参考:

```yaml
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## 解除注释
  log_keys:
    - txt'

  lora_config: ##  解除注释
    target: sat.model.finetune.lora2.LoraMixin
    params:
      r: 256
```

### 修改运行脚本

编辑`finetune_single_gpu.sh` 或者 `finetune_multi_gpus.sh`，选择配置文件。下面是两个例子:

1. 如果您想使用 `CogVideoX-2B` 模型并使用`Lora`方案，您需要修改`finetune_single_gpu.sh` 或者 `finetune_multi_gpus.sh`:

```
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
```

2. 如果您想使用 `CogVideoX-2B` 模型并使用`全量微调`方案，您需要修改`finetune_single_gpu.sh`
   或者 `finetune_multi_gpus.sh`:

```
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b.yaml configs/sft.yaml --seed $RANDOM"
```

### 微调和验证

运行推理代码,即可开始微调。

```shell
bash finetune_single_gpu.sh # Single GPU
bash finetune_multi_gpus.sh # Multi GPUs
```

### 使用微调后的模型

微调后的模型无法合并，这里展现了如何修改推理配置文件 `inference.sh`

```
run_cmd="$environs python sample_video.py --base configs/cogvideox_<模型参数>_lora.yaml configs/inference.yaml --seed 42"
```

然后，执行代码:

```
bash inference.sh 
```

### 转换到 Huggingface Diffusers 库支持的权重

SAT 权重格式与 Huggingface 的权重格式不同，需要转换。请运行

```shell
python ../tools/convert_weight_sat2hf.py
```

### 从SAT权重文件 导出Huggingface Diffusers lora权重

支持了从SAT权重文件
在经过上面这些步骤训练之后，我们得到了一个sat带lora的权重，在{args.save}/1000/1000/mp_rank_00_model_states.pt你可以看到这个文件

导出的lora权重脚本在CogVideoX仓库 tools/export_sat_lora_weight.py ,导出后使用 load_cogvideox_lora.py 推理

导出命令:

```
python tools/export_sat_lora_weight.py --sat_pt_path {args.save}/{experiment_name}-09-09-21-10/1000/mp_rank_00_model_states.pt --lora_save_directory   {args.save}/export_hf_lora_weights_1/
```

这次训练主要修改了下面几个模型结构,下面列出了 转换为HF格式的lora结构对应关系,可以看到lora将模型注意力结构上增加一个低秩权重,

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

通过export_sat_lora_weight.py将它转换为HF格式的lora结构
![alt text](../resources/hf_lora_weights.png)
