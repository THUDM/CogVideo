# SAT CogVideoX-2B

This folder contains the inference code using [SAT](https://github.com/THUDM/SwissArmyTransformer) weights and the
fine-tuning code for SAT weights.

This code is the framework used by the team to train the model. It has few comments and requires careful study.

## Inference Model

1. Ensure that you have correctly installed the dependencies required by this folder.

```shell
pip install -r requirements.txt
```

2. Download the model weights

First, go to the SAT mirror to download the dependencies.

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

Then unzip, the model structure should look like this:

```
.
├── transformer
│   ├── 1000
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

Next, clone the T5 model, which is not used for training and fine-tuning, but must be used.

```
git clone https://huggingface.co/THUDM/CogVideoX-2b.git
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

By following the above approach, you will obtain a safetensor format T5 file. Ensure that there are no errors when
loading it into Deepspeed in Finetune.

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

3. Modify the file `configs/cogvideox_2b_infer.yaml`.

```yaml
load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer model path

conditioner_config:
  target: sgm.modules.GeneralConditioner
  params:
    emb_models:
      - is_trainable: false
        input_key: txt
        ucg_rate: 0.1
        target: sgm.modules.encoders.modules.FrozenT5Embedder
        params:
          model_dir: "google/t5-v1_1-xxl" ## T5 model path
          max_length: 226

first_stage_config:
  target: sgm.models.autoencoder.VideoAutoencoderInferenceWrapper
  params:
    cp_size: 1
    ckpt_path: "{your_CogVideoX-2b-sat_path}/vae/3d-vae.pt" ## VAE model path
```

+ If using txt to save multiple prompts, please refer to `configs/test.txt` for modification. One prompt per line. If
  you don't know how to write prompts, you can first use [this code](../inference/convert_demo.py) to call LLM for
  refinement.
+ If using the command line as input, modify

```yaml
input_type: cli
```

so that prompts can be entered from the command line.

If you want to change the output video directory, you can modify:

```yaml
output_dir: outputs/
```

The default is saved in the `.outputs/` folder.

4. Run the inference code to start inference

```shell
bash inference.sh
```

## Fine-Tuning the Model

### Preparing the Environment

Please note that currently, SAT needs to be installed from the source code for proper fine-tuning.

You need to get the code from the source to support the fine-tuning functionality, as these features have not yet been
released in the Pip package.

We will address this issue in future stable releases.


```
git clone https://github.com/THUDM/SwissArmyTransformer.git
cd SwissArmyTransformer
pip install -e .
```

### Preparing the Dataset

The dataset format should be as follows:

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

Each txt file should have the same name as its corresponding video file and contain the labels for that video. Each
video should have a one-to-one correspondence with a label. Typically, a video should not have multiple labels.

For style fine-tuning, please prepare at least 50 videos and labels with similar styles to facilitate fitting.

### Modifying the Configuration File

We support both `Lora` and `full-parameter fine-tuning` methods. Please note that both fine-tuning methods only apply to
the `transformer` part. The `VAE part` is not modified. `T5` is only used as an Encoder.

the `configs/cogvideox_2b_sft.yaml` (for full fine-tuning) as follows.

```yaml
  # checkpoint_activations: True ## using gradient checkpointing (both checkpoint_activations in the configuration file need to be set to True)
  model_parallel_size: 1 # Model parallel size
  experiment_name: lora-disney  # Experiment name (do not change)
  mode: finetune # Mode (do not change)
  load: "{your_CogVideoX-2b-sat_path}/transformer" # Transformer model path
  no_load_rng: True # Whether to load the random seed
  train_iters: 1000 # Number of training iterations
  eval_iters: 1 # Number of evaluation iterations
  eval_interval: 100 # Evaluation interval
  eval_batch_size: 1 # Batch size for evaluation
  save: ckpts # Model save path
  save_interval: 100 # Model save interval
  log_interval: 20 # Log output interval
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # Training and validation sets can be the same
  split: 1,0,0 # Ratio of training, validation, and test sets
  num_workers: 8 # Number of worker threads for data loading
  force_train: True # Allow missing keys when loading ckpt (refer to T5 and VAE which are loaded independently)
  only_log_video_latents: True # Avoid using VAE decoder when eval to save memory
```

If you wish to use Lora fine-tuning, you also need to modify:

```yaml
model:
  scale_factor: 1.15258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## Uncomment
  log_keys:
    - txt'

  lora_config: ## Uncomment
    target: sat.model.finetune.lora2.LoraMixin
    params:
      r: 256
```

### Fine-Tuning and Validation

1. Run the inference code to start fine-tuning.

```shell
bash finetune_single_gpu.sh # Single GPU
bash finetune_multi_gpus.sh # Multi GPUs
```

### Converting to Huggingface Diffusers Supported Weights

The SAT weight format is different from Huggingface's weight format and needs to be converted. Please run:

```shell
python ../tools/convert_weight_sat2hf.py
```

**Note**: This content has not yet been tested with LORA fine-tuning models.