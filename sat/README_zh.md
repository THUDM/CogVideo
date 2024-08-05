# SAT CogVideoX-2B

本文件夹包含了使用 [SAT](https://github.com/THUDM/SwissArmyTransformer) 权重的推理代码，以及 SAT 权重的微调代码。

该代码是团队训练模型时使用的框架。注释较少，需要认真研究。

## 推理模型

1. 确保你已经正确安装本文件夹中的要求的依赖

```shell
pip install -r requirements.txt
```

2. 下载模型权重

首先，前往 SAT 镜像下载依赖。

```shell
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
uzip vae.zip
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
```

然后，解压文件，模型结构应该如下

```
.
├── transformer
│   ├── 1000
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

接着，克隆 T5 模型，该模型不用做训练和微调，但是必须使用。

```shell
git lfs install 
git clone https://huggingface.co/google/t5-v1_1-xxl.git
```

**我们不需要使用tf_model.h5**文件。该文件可以删除。

3. 修改`configs/cogvideox_2b_infer.yaml`中的文件。

```yaml
load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer 模型路径

conditioner_config:
  target: sgm.modules.GeneralConditioner
  params:
    emb_models:
      - is_trainable: false
        input_key: txt
        ucg_rate: 0.1
        target: sgm.modules.encoders.modules.FrozenT5Embedder
        params:
          model_dir: "google/t5-v1_1-xxl" ## T5 模型路径
          max_length: 226

first_stage_config:
  target: sgm.models.autoencoder.VideoAutoencoderInferenceWrapper
  params:
    cp_size: 1
    ckpt_path: "{your_CogVideoX-2b-sat_path}/vae/3d-vae.pt" ## VAE 模型路径

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

4. 运行推理代码,即可推理

```shell
bash inference.sh
```

## 微调模型

### 准备数据集

数据集格式应该如下：

```
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
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
部分。 请按照以下方式修改`configs/cogvideox_2b_sft.yaml`(全量微调) 中的文件。

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
```

如果你希望使用 Lora 微调，你还需要修改：

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

### 微调和验证

1. 运行推理代码,即可开始微调。

```shell
bash finetune.sh
```

### 转换到 Huggingface Diffusers 库支持的权重

SAT 权重格式与 Huggingface 的权重格式不同，需要转换。请运行

```shell
python ../tools/convert_weight_sat2hf.py
```

**注意** 本内容暂未测试 LORA 微调模型。