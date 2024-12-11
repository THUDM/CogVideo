# SAT CogVideoX

[中文阅读](./README_zh.md)

[日本語で読む](./README_ja.md)

This folder contains inference code using [SAT](https://github.com/THUDM/SwissArmyTransformer) weights, along with fine-tuning code for SAT weights. 

This code framework was used by our team during model training. There are few comments, so careful study is required.

## Inference Model

### 1. Make sure you have installed all dependencies in this folder

```
pip install -r requirements.txt
```

### 2. Download the Model Weights

First, download the model weights from the SAT mirror.

#### CogVideoX1.5 Model

```
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT
```

This command downloads three models: Transformers, VAE, and T5 Encoder.

#### CogVideoX Model

For the CogVideoX-2B model, download as follows:

```
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
```

Download the `transformers` file for the CogVideoX-5B model (the VAE file is the same as for 2B):

+ [CogVideoX-5B](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX-5B-I2V](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)

Arrange the model files in the following structure:

```
.
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

Since model weight files are large, it’s recommended to use `git lfs`.  
See [here](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing) for `git lfs` installation.

```
git lfs install
```

Next, clone the T5 model, which is used as an encoder and doesn’t require training or fine-tuning.
> You may also use the model file location on [Modelscope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b).

```
git clone https://huggingface.co/THUDM/CogVideoX-2b.git # Download model from Huggingface
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git # Download from Modelscope
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

This will yield a safetensor format T5 file that can be loaded without error during Deepspeed fine-tuning.

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

### 3. Modify `configs/cogvideox_*.yaml` file.

```yaml
model:
  scale_factor: 1.55258426
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
            model_dir: "t5-v1_1-xxl" # absolute path to CogVideoX-2b/t5-v1_1-xxl weight folder
            max_length: 226

  first_stage_config:
    target: vae_modules.autoencoder.VideoAutoencoderInferenceWrapper
    params:
      cp_size: 1
      ckpt_path: "CogVideoX-2b-sat/vae/3d-vae.pt" # absolute path to CogVideoX-2b-sat/vae/3d-vae.pt file
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

### 4. Modify `configs/inference.yaml` file.

```yaml
args:
  latent_channels: 16
  mode: inference
  load: "{absolute_path/to/your}/transformer" # Absolute path to CogVideoX-2b-sat/transformer folder
  # load: "{your lora folder} such as zRzRzRzRzRzRzR/lora-disney-08-20-13-28" # This is for Full model without lora adapter

  batch_size: 1
  input_type: txt # You can choose "txt" for plain text input or change to "cli" for command-line input
  input_file: configs/test.txt # Plain text file, can be edited
  sampling_num_frames: 13  # For CogVideoX1.5-5B it must be 42 or 22. For CogVideoX-5B / 2B, it must be 13, 11, or 9.
  sampling_fps: 8
  fp16: True # For CogVideoX-2B
  # bf16: True # For CogVideoX-5B
  output_dir: outputs/
  force_inference: True
```

+ If using a text file to save multiple prompts, modify `configs/test.txt` as needed. One prompt per line. If you are unsure how to write prompts, use [this code](../inference/convert_demo.py) to call an LLM for refinement.
+ To use command-line input, modify:

```
input_type: cli
```

This allows you to enter prompts from the command line.

To modify the output video location, change:

```
output_dir: outputs/
```

The default location is the `.outputs/` folder.

### 5. Run the Inference Code to Perform Inference

```
bash inference.sh
```

## Fine-tuning the Model

### Preparing the Dataset

The dataset should be structured as follows:

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

Each txt file should have the same name as the corresponding video file and contain the label for that video. The videos and labels should correspond one-to-one. Generally, avoid using one video with multiple labels.

For style fine-tuning, prepare at least 50 videos and labels with a similar style to facilitate fitting.

### Modifying the Configuration File

We support two fine-tuning methods: `Lora` and full-parameter fine-tuning. Note that both methods only fine-tune the `transformer` part. The `VAE` part is not modified, and `T5` is only used as an encoder.
Modify the files in `configs/sft.yaml` (full fine-tuning) as follows:

```yaml
  # checkpoint_activations: True ## using gradient checkpointing (both `checkpoint_activations` in the config file need to be set to True)
  model_parallel_size: 1 # Model parallel size
  experiment_name: lora-disney  # Experiment name (do not change)
  mode: finetune # Mode (do not change)
  load: "{your_CogVideoX-2b-sat_path}/transformer" ## Path to Transformer model
  no_load_rng: True # Whether to load random number seed
  train_iters: 1000 # Training iterations
  eval_iters: 1 # Evaluation iterations
  eval_interval: 100    # Evaluation interval
  eval_batch_size: 1  # Evaluation batch size
  save: ckpts # Model save path 
  save_interval: 100 # Save interval
  log_interval: 20 # Log output interval
  train_data: [ "your train data path" ]
  valid_data: [ "your val data path" ] # Training and validation sets can be the same
  split: 1,0,0 # Proportion for training, validation, and test sets
  num_workers: 8 # Number of data loader workers
  force_train: True # Allow missing keys when loading checkpoint (T5 and VAE loaded separately)
  only_log_video_latents: True # Avoid memory usage from VAE decoding
  deepspeed:
    bf16:
      enabled: False # For CogVideoX-2B Turn to False and For CogVideoX-5B Turn to True
    fp16:
      enabled: True  # For CogVideoX-2B Turn to True and For CogVideoX-5B Turn to False
```

``` To use Lora fine-tuning, you also need to modify `cogvideox_<model parameters>_lora` file:

Here's an example using `CogVideoX-2B`:

```
model:
  scale_factor: 1.55258426
  disable_first_stage_autocast: true
  not_trainable_prefixes: [ 'all' ] ## Uncomment to unlock
  log_keys:
    - txt

  lora_config: ## Uncomment to unlock
    target: sat.model.finetune.lora2.LoraMixin
    params:
      r: 256
```

### Modify the Run Script

Edit `finetune_single_gpu.sh` or `finetune_multi_gpus.sh` and select the config file. Below are two examples:

1. If you want to use the `CogVideoX-2B` model with `Lora`, modify `finetune_single_gpu.sh` or `finetune_multi_gpus.sh` as follows:

```
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
```

2. If you want to use the `CogVideoX-2B` model with full fine-tuning, modify `finetune_single_gpu.sh` or `finetune_multi_gpus.sh` as follows:

```
run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_2b.yaml configs/sft.yaml --seed $RANDOM"
```

### Fine-tuning and Validation

Run the inference code to start fine-tuning.

```
bash finetune_single_gpu.sh # Single GPU
bash finetune_multi_gpus.sh # Multi GPUs
```

### Using the Fine-tuned Model

The fine-tuned model cannot be merged. Here’s how to modify the inference configuration file `inference.sh`

```
run_cmd="$environs python sample_video.py --base configs/cogvideox_<model parameters>_lora.yaml configs/inference.yaml --seed 42"
```

Then, run the code:

```
bash inference.sh 
```

### Converting to Huggingface Diffusers-compatible Weights

The SAT weight format is different from Huggingface’s format and requires conversion. Run

```
python ../tools/convert_weight_sat2hf.py
```

### Exporting Lora Weights from SAT to Huggingface Diffusers

Support is provided for exporting Lora weights from SAT to Huggingface Diffusers format.
 After training with the above steps, you’ll find the SAT model with Lora weights in {args.save}/1000/1000/mp_rank_00_model_states.pt

The export script `export_sat_lora_weight.py` is located in the CogVideoX repository under `tools/`. After exporting, use `load_cogvideox_lora.py` for inference.

Export command:

```
python tools/export_sat_lora_weight.py --sat_pt_path {args.save}/{experiment_name}-09-09-21-10/1000/mp_rank_00_model_states.pt --lora_save_directory   {args.save}/export_hf_lora_weights_1/
```

The following model structures were modified during training. Here is the mapping between SAT and HF Lora structures. Lora adds a low-rank weight to the attention structure of the model.

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

Using `export_sat_lora_weight.py` will convert these to the HF format Lora structure.
![alt text](../resources/hf_lora_weights.png)