# ビデオキャプション

通常、ほとんどのビデオデータには対応する説明文が付いていないため、ビデオデータをテキストの説明に変換して、テキストからビデオへのモデルに必要なトレーニングデータを提供する必要があります。

## CogVLM2-Video を使用したビデオキャプション

<p align="center">
🤗 <a href="https://huggingface.co/THUDM/cogvlm2-video-llama3-chat">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://cogvlm2-video.github.io/">ブログ</a> &nbsp&nbsp ｜ <a href="http://cogvlm2-online.cogviewai.cn:7868/">💬 オンラインデモ</a>&nbsp&nbsp
</p>

CogVLM2-Video は、タイムスタンプベースの質問応答機能を備えた多機能なビデオ理解モデルです。ユーザーは `このビデオを詳細に説明してください。` などのプロンプトをモデルに入力して、詳細なビデオキャプションを取得できます：
<div align="center">
    <a href="https://cogvlm2-video.github.io/"><img width="600px" height="auto" src="./assests/cogvlm2-video-example.png"></a>
</div>

ユーザーは提供された[コード](https://github.com/THUDM/CogVLM2/tree/main/video_demo)を使用してモデルをロードするか、RESTful API を構成してビデオキャプションを生成できます。
