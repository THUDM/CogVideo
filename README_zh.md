# CogVideo && CogVideoX

[Read this in English.](./README_zh)

[日本語で読む](./README_ja.md)


<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
🤗 在 <a href="https://huggingface.co/spaces/THUDM/CogVideoX" target="_blank">CogVideoX Huggingface Space</a> 体验视频生成模型
</p>
<p align="center">
📚 查看 <a href="https://arxiv.org/abs/2408.06072" target="_blank">论文</a>
</p>
<p align="center">
    👋 加入我们的 <a href="resources/WECHAT.md" target="_blank">微信</a> 和  <a href="https://discord.gg/Ewaabk6s" target="_blank">Discord</a> 
</p>
<p align="center">
📍 前往<a href="https://chatglm.cn/video?fr=osm_cogvideox"> 清影</a> 和 <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9"> API平台</a> 体验更大规模的商业版视频生成模型。
</p>

## 项目更新

- 🔥🔥 **News**: ```2024/8/12```: CogVideoX 论文已上传到arxiv，欢迎查看[论文](https://arxiv.org/abs/2408.06072)。
- 🔥 **News**: ```2024/8/7```: CogVideoX 已经合并入 `diffusers`
  0.30.0版本，单张3090可以推理，详情请见[代码](inference/cli_demo.py)。
- 🔥 **News**: ```2024/8/6```: 我们开源 **3D Causal VAE**，用于 **CogVideoX-2B**，可以几乎无损地重构视频。
- 🔥 **News**: ```2024/8/6```: 我们开源 CogVideoX 系列视频生成模型的第一个模型, **CogVideoX-2B**。
- 🌱 **Source**: ```2022/5/19```: 我们开源了 CogVideo 视频生成模型（现在你可以在 `CogVideo` 分支中看到），这是首个开源的基于
  Transformer 的大型文本生成视频模型，您可以访问 [ICLR'23 论文](https://arxiv.org/abs/2205.15868) 查看技术细节。
  **性能更强，参数量更大的模型正在到来的路上～，欢迎关注**

## 目录

跳转到指定部分：

- [快速开始](#快速开始)
    - [SAT](#sat)
    - [Diffusers](#Diffusers)
- [CogVideoX-2B 视频作品](#cogvideox-2b-视频作品)
- [CogVideoX模型介绍](#模型介绍)
- [完整项目代码结构](#完整项目代码结构)
    - [Inference](#inference)
    - [SAT](#sat)
    - [Tools](#tools)
- [开源项目规划](#开源项目规划)
- [模型协议](#模型协议)
- [CogVideo(ICLR'23)模型介绍](#cogvideoiclr23)
- [引用](#引用)

## 快速开始

### 提示词优化

在开始运行模型之前，请参考[这里](inference/convert_demo.py) 查看我们是怎么使用GLM-4(或者同级别的其他产品，例如GPT-4)
大模型对模型进行优化的，这很重要，
由于模型是在长提示词下训练的，一个好的提示词直接影响了视频生成的质量。

### SAT

查看sat文件夹下的[sat_demo](sat/README.md)：包含了 SAT 权重的推理代码和微调代码，推荐基于此代码进行 CogVideoX
模型结构的改进，研究者使用该代码可以更好的进行快速的迭代和开发。
(18 GB 推理, 40GB lora微调)

### Diffusers

```
pip install -r requirements.txt
```

查看[diffusers_demo](inference/cli_demo.py)：包含对推理代码更详细的解释，包括各种关键的参数。（24GB 推理，微调代码正在开发）

## CogVideoX-2B 视频作品

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

## 模型介绍

CogVideoX是 [清影](https://chatglm.cn/video?fr=osm_cogvideox) 同源的开源版本视频生成模型。

下表展示目前我们提供的视频生成模型列表，以及相关基础信息:

| 模型名                 | CogVideoX-2B                                                                                                                    | 
|---------------------|---------------------------------------------------------------------------------------------------------------------------------|
| 提示词语言               | English                                                                                                                         | 
| 单GPU推理 (FP-16) 显存消耗 | 18GB using [SAT](https://github.com/THUDM/SwissArmyTransformer)   <br>  23.9GB using diffusers                                  | 
| 多GPU推理 (FP-16) 显存消耗 | 20GB minimum per GPU using diffusers                                                                                            |                                                                                                            
| 微调显存消耗 (bs=1)       | 42GB                                                                                                                            |
| 提示词长度上限             | 226 Tokens                                                                                                                      |
| 视频长度                | 6 seconds                                                                                                                       | 
| 帧率（每秒）              | 8 frames                                                                                                                        | 
| 视频分辨率               | 720 * 480                                                                                                                       |
| 量化推理                | 不支持                                                                                                                             |          
| 下载地址 (Diffusers 模型) | 🤗 [Huggingface](https://huggingface.co/THUDM/CogVideoX-2B)  [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b) |
| 下载地址 (SAT 模型)       | [SAT](./sat/README_zh.md)                                                                                                       |

## 友情链接

我们非常欢迎来自社区的贡献，并积极的贡献开源社区。以下作品已经对CogVideoX进行了适配，欢迎大家使用:

+ [Xorbits Inference](https://github.com/xorbitsai/inference): 性能强大且功能全面的分布式推理框架，轻松一键部署你自己的模型或内置的前沿开源模型。

## 完整项目代码结构

本开源仓库将带领开发者快速上手 **CogVideoX** 开源模型的基础调用方式、微调示例。

### inference

+ [diffusers_demo](inference/cli_demo.py): 更详细的推理代码讲解，常见参数的意义，在这里都会提及。
+ [diffusers_vae_demo](inference/cli_vae_demo.py): 单独执行VAE的推理代码，目前需要71GB显存，将来会优化。
+ [convert_demo](inference/convert_demo.py): 如何将用户的输入转换成适合
  CogVideoX的长输入。因为CogVideoX是在长文本上训练的，所以我们需要把输入文本的分布通过LLM转换为和训练一致的长文本。脚本中默认使用GLM4，也可以替换为GPT、Gemini等任意大语言模型。
+ [gradio_web_demo](inference/gradio_web_demo.py): 一个简单的gradio网页应用，展示如何使用 CogVideoX-2B 模型生成视频。
  与我们的 Huggingface Space 类似，你可以使用此脚本运行一个简单的网页应用，用于生成视频。

```shell
cd inference
# For Linux and Windows users (and macOS with Intel??)
python gradio_web_demo.py # humans mode

# For macOS with Apple Silicon users, Intel not supported, this maybe 20x slower than RTX 4090
PYTORCH_ENABLE_MPS_FALLBACK=1 python gradio_web_demo.py # humans mode
```

<div style="text-align: center;">
    <img src="resources/gradio_demo.png" style="width: 100%; height: auto;" />
</div>

+ [streamlit_web_demo](inference/streamlit_web_demo.py): 一个简单的streamlit网页应用，展示如何使用 CogVideoX-2B 模型生成视频。

<div style="text-align: center;">
    <img src="resources/web_demo.png" style="width: 100%; height: auto;" />
</div>

### sat

+ [sat_demo](sat/README_zh.md): 包含了 SAT 权重的推理代码和微调代码，推荐基于 CogVideoX
  模型结构进行改进，创新的研究者使用改代码以更好的进行快速的堆叠和开发。

### tools

本文件夹包含了一些工具，用于模型的转换 / Caption 等工作。

+ [convert_weight_sat2hf](tools/convert_weight_sat2hf.py): 将 SAT 模型权重转换为 Huggingface 模型权重。
+ [caption_demo](tools/caption/README_zh.md):  Caption 工具，对视频理解并用文字输出的模型。

## CogVideo(ICLR'23)

[CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)
的官方repo位于[CogVideo branch](https://github.com/THUDM/CogVideo/tree/CogVideo)。

**CogVideo可以生成高帧率视频，下面展示了一个32帧的4秒视频。**

![High-frame-rate sample](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/appendix-sample-highframerate.png)

![Intro images](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/intro-image.png)


<div align="center">
  <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="80%" controls autoplay></video>
</div>

CogVideo的demo网站在[https://models.aminer.cn/cogvideo](https://models.aminer.cn/cogvideo/)。您可以在这里体验文本到视频生成。
*原始输入为中文。*

## 引用

🌟 如果您发现我们的工作有所帮助，欢迎引用我们的文章，留下宝贵的stars

```
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
@article{hong2022cogvideo,
  title={CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers},
  author={Hong, Wenyi and Ding, Ming and Zheng, Wendi and Liu, Xinghan and Tang, Jie},
  journal={arXiv preprint arXiv:2205.15868},
  year={2022}
}
```

## 开源项目规划

- [x] CogVideoX 模型开源
    - [x] CogVideoX 模型推理示例 (CLI / Web Demo)
    - [x] CogVideoX 在线体验示例 (Huggingface Space)
    - [x] CogVideoX 开源模型API接口示例 (Huggingface)
    - [x] CogVideoX 模型微调示例 (SAT)
    - [ ] CogVideoX 模型微调示例 (Huggingface / SAT)
    - [ ] CogVideoX-Pro 开源(适配 CogVideoX-2B 套件)
    - [X] CogVideoX 技术报告公开

我们欢迎您的贡献，您可以点击[这里](resources/contribute_zh.md)查看更多信息。

## 模型协议

本仓库代码使用 [Apache 2.0 协议](LICENSE) 发布。

本模型权重和模型实现代码根据 [CogVideoX LICENSE](MODEL_LICENSE) 许可证发布。
