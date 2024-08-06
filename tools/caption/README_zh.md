# 视频Caption

通常，大多数视频数据不带有相应的描述性文本，因此需要将视频数据转换为文本描述，以提供必要的训练数据用于文本到视频模型。

## 通过 CogVLM2-Video 模型生成视频Caption

🤗 [Hugging Face](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat) | 🤖 [ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat) | 📑 [Blog](https://cogvlm2-video.github.io/) ｜ [💬 Online Demo](http://cogvlm2-online.cogviewai.cn:7868/)

CogVLM2-Video 是一个多功能的视频理解模型，具备基于时间戳的问题回答能力。用户可以输入诸如 `请详细描述这个视频` 的提示语给模型，以获得详细的视频Caption：


<div align="center">
    <a href="https://cogvlm2-video.github.io/"><img width="600px" height="auto" src="./assests/cogvlm2-video-example.png"></a>
</div>

用户可以使用提供的[代码](https://github.com/THUDM/CogVLM2/tree/main/video_demo)加载模型或配置 RESTful API 来生成视频Caption。