# Video Caption

Typically, most video data does not come with corresponding descriptive text, so it is necessary to convert the video
data into textual descriptions to provide the essential training data for text-to-video models.

## Video Caption via CogVLM2-Video

<p align="center">
🤗 <a href="https://huggingface.co/THUDM/cogvlm2-video-llama3-chat">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://cogvlm2-video.github.io/">Blog</a> &nbsp&nbsp ｜ <a href="http://cogvlm2-online.cogviewai.cn:7868/">💬 Online Demo</a>&nbsp&nbsp
</p>

CogVLM2-Video is a versatile video understanding model equipped with timestamp-based question answering capabilities.
Users can input prompts such as `Please describe this video in detail.` to the model to obtain a detailed video caption:
<div align="center">
    <a href="https://cogvlm2-video.github.io/"><img width="600px" height="auto" src="./assests/cogvlm2-video-example.png"></a>
</div>

Users can use the provided [code](https://github.com/THUDM/CogVLM2/tree/main/video_demo) to load the model or configure a RESTful API to generate video captions.