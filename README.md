# CogVideoX

[中文阅读](./README_zh.md)

<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
🤗 Experience on <a href="https://huggingface.co/spaces/THUDM/CogVideoX" target="_blank">CogVideoX Huggingface Space</a>
</p>
<p align="center">
📚 Check here to view <a href="resources/CogVideoX.pdf" target="_blank">Paper</a>
</p>
<p align="center">
    👋 Join our <a href="resources/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/Ewaabk6s" target="_blank">Discord</a> 
</p>
<p align="center">
📍 Visit <a href="https://chatglm.cn/video">清影</a> and <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">API Platform</a> to experience larger-scale commercial video generation models.
</p>

## Update and News

- 🔥 **News**: ``2024/8/6``: We have also open-sourced **3D Causal VAE** used in **CogVideoX-2B**, which can reconstruct
  the video almost losslessly.
- 🔥 **News**: ``2024/8/6``: We have open-sourced **CogVideoX-2B**，the first model in the CogVideoX series of video
  generation models.

**More powerful models with larger parameter sizes are on the way~ Stay tuned!**

## CogVideoX-2B Gallery

<div align="center">
  <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="80%" controls autoplay></video>
  <p>A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.</p>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/9de41efd-d4d1-4095-aeda-246dd834e91d" width="80%" controls autoplay></video>
  <p>The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from its tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.</p>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/941d6661-6a8d-4a1b-b912-59606f0b2841" width="80%" controls autoplay></video>
  <p>A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall.</p>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/938529c4-91ae-4f60-b96b-3c3947fa63cb" width="80%" controls autoplay></video>
  <p>In the haunting backdrop of a war-torn city, where ruins and crumbled walls tell a story of devastation, a poignant close-up frames a young girl. Her face is smudged with ash, a silent testament to the chaos around her. Her eyes glistening with a mix of sorrow and resilience, capturing the raw emotion of a world that has lost its innocence to the ravages of conflict.</p>
</div>

## Model Introduction

CogVideoX is an open-source version of the video generation model, which is homologous
to [清影](https://chatglm.cn/video).

The table below shows the list of video generation models we currently provide,
along with related basic information:

| Model Name                                | CogVideoX-2B                                                                                                                         | 
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Prompt Language                           | English                                                                                                                              | 
| GPU Memory Required for Inference (FP16)  | 36GB using diffusers (will be optimized before the PR is merged)  and 25G using [SAT](https://github.com/THUDM/SwissArmyTransformer) | 
| GPU Memory Required for Fine-tuning(bs=1) | 42GB                                                                                                                                 |
| Prompt Max  Length                        | 226 Tokens                                                                                                                           |
| Video Length                              | 6 seconds                                                                                                                            | 
| Frames Per Second                         | 8 frames                                                                                                                             | 
| Resolution                                | 720 * 480                                                                                                                            |
| Quantized Inference                       | Not Supported                                                                                                                        |          
| Multi-card Inference                      | Not Supported                                                                                                                        |                             
| Download Link                             | 🤗 [CogVideoX-2B](https://huggingface.co/THUDM/CogVideoX-2B)                                                                         |

## Project Structure

This open-source repository will guide developers to quickly get started with the basic usage and fine-tuning examples
of the **CogVideoX** open-source model.

### Inference

+ [cli_demo](inference/cli_demo.py): A more detailed explanation of the inference code, mentioning the significance of
  common parameters.
+ [cli_vae_demo](inference/cli_vae_demo.py): Executing the VAE inference code alone currently requires 71GB of memory,
  but it will be optimized in the future.
+ [convert_demo](inference/converter_demo.py): How to convert user input into a format suitable for CogVideoX.
+ [web_demo](inference/web_demo.py): A simple streamlit web application demonstrating how to use the CogVideoX-2B model
  to generate videos.

<div style="text-align: center;">
    <img src="resources/web_demo.png" style="width: 100%; height: auto;" />
</div>

### sat

+ [sat_demo](sat/README.md): Contains the inference code and fine-tuning code of SAT weights. It is
  recommended to improve based on the CogVideoX model structure. Innovative researchers use this code to better perform
  rapid stacking and development.

### Tools

This folder contains some tools for model conversion / caption generation, etc.

+ [convert_weight_sat2hf](tools/convert_weight_sat2hf.py): Convert SAT model weights to Huggingface model weights.
+ [caption_demo](tools/caption_demo.py): Caption tool, a model that understands videos and outputs them in text.

## Project Plan

- [x] Open source CogVideoX model
    - [x] Open source 3D Causal VAE used in CogVideoX.
    - [x] CogVideoX model inference example (CLI / Web Demo)
    - [x] CogVideoX online experience demo (Huggingface Space)
    - [x] CogVideoX open source model API interface example (Huggingface)
    - [x] CogVideoX model fine-tuning example (SAT)
    - [ ] CogVideoX model fine-tuning example (Huggingface / SAT)
    - [ ] Open source CogVideoX-Pro (adapted for CogVideoX-2B suite)
    - [ ] Release CogVideoX technical report

We welcome your contributions. You can click [here](resources/contribute.md) for more information.

## Model License

The code in this repository is released under the [Apache 2.0 License](LICENSE).

The model weights and implementation code are released under the [CogVideoX LICENSE](MODEL_LICENSE).

## Citation

🌟 If you find our work helpful, please leave us a star. 🌟

The paper is still being written and will be released soon. Stay tuned!