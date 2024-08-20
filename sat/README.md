# SAT CogVideoX-2B

[中文阅读](./README_zh.md)

[日本語で読む](./README_ja.md)

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

Each text file shares the same name as its corresponding video, serving as the label for that video. Videos and labels
should be matched one-to-one. Generally, a single video should not be associated with multiple labels.

For style fine-tuning, please prepare at least 50 videos and labels with similar styles to ensure proper fitting.

### Modifying Configuration Files

We support two fine-tuning methods: `Lora` and full-parameter fine-tuning. Please note that both methods only fine-tune
the `transformer` part and do not modify the `VAE` section. `T5` is used solely as an Encoder. Please modify
the `configs/sft.yaml` (for full-parameter fine-tuning) file as follows:

```
  # checkpoint_activations: True ## Using gradient checkpointing (Both checkpoint_activations in the config file need to be set to True)
  model_parallel_size: 1 # Model parallel size
  experiment_name: lora-disney  # Experiment name (do not modify)
  mode: finetune # Mode (do not modify)
  load: "{your_CogVideoX-2b-sat_path}/transformer" ## Transformer model path
  no_load_rng: True # Whether to load random seed
  train_iters: 1000 # Training iterations
  eval_iters: 1 # Evaluation iterations
  eval_interval: 100    # Evaluation interval
  eval_batch_size: 1  # Evaluation batch size
  save: ckpts # Model save path
  save_interval: 100 # Model save interval
  log_interval: 20 # Log output interval
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # Training and validation datasets can be the same
  split: 1,0,0 # Training, validation, and test set ratio
  num_workers: 8 # Number of worker threads for data loader
  force_train: True # Allow missing keys when loading checkpoint (T5 and VAE are loaded separately)
  only_log_video_latents: True # Avoid memory overhead caused by VAE decode
  deepspeed:
    bf16:
      enabled: False # For CogVideoX-2B set to False and for CogVideoX-5B set to True
    fp16:
      enabled: True  # For CogVideoX-2B set to True and for CogVideoX-5B set to False
```

If you wish to use Lora fine-tuning, you also need to modify the `cogvideox_<model_parameters>_lora` file:

Here, take `CogVideoX-2B` as a reference:

```
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

### Modifying Run Scripts

Edit `finetune_single_gpu.sh` or `finetune_multi_gpus.sh` to select the configuration file. Below are two examples:

1. If you want to use the `CogVideoX-2B` model and the `Lora` method, you need to modify `finetune_single_gpu.sh`
   or `finetune_multi_gpus.sh`:

```
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
```

2. If you want to use the `CogVideoX-2B` model and the `full-parameter fine-tuning` method, you need to
   modify `finetune_single_gpu.sh` or `finetune_multi_gpus.sh`:

```
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b.yaml configs/sft.yaml --seed $RANDOM"
```

### Fine-Tuning and Evaluation

Run the inference code to start fine-tuning.

```
bash finetune_single_gpu.sh # Single GPU
bash finetune_multi_gpus.sh # Multi GPUs
```

### Using the Fine-Tuned Model

The fine-tuned model cannot be merged; here is how to modify the inference configuration file `inference.sh`:

```
run_cmd="$environs python sample_video.py --base configs/cogvideox_<model_parameters>_lora.yaml configs/inference.yaml --seed 42"
```

Then, execute the code:

```
bash inference.sh 
```

### Converting to Huggingface Diffusers Supported Weights

The SAT weight format is different from Huggingface's weight format and needs to be converted. Please run:

```shell
python ../tools/convert_weight_sat2hf.py
```

**Note**: This content has not yet been tested with LORA fine-tuning models.