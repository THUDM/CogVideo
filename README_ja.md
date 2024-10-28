# CogVideo & CogVideoX

[Read this in English](./README_zh.md)

[中文阅读](./README_zh.md)

<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
<a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B" target="_blank"> 🤗 Huggingface Space</a> または <a href="https://modelscope.cn/studios/ZhipuAI/CogVideoX-5b-demo" target="_blank"> 🤖 ModelScope Space</a> で CogVideoX-5B モデルをオンラインで体験してください
</p>
<p align="center">
📚 <a href="https://arxiv.org/abs/2408.06072" target="_blank">論文</a>と<a href="https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh" target="_blank">使用ドキュメント</a>を表示します。
</p>
<p align="center">
    👋 <a href="resources/WECHAT.md" target="_blank">WeChat</a> と <a href="https://discord.gg/dCGfUsagrD" target="_blank">Discord</a> に参加
</p>
<p align="center">
📍 <a href="https://chatglm.cn/video?lang=en?fr=osm_cogvideo">清影</a> と <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">APIプラットフォーム</a> を訪問して、より大規模な商用ビデオ生成モデルを体験.
</p>

## 更新とニュース

- 🔥🔥 **ニュース**: ```2024/10/13```: コスト削減のため、単一の4090 GPUで`CogVideoX-5B`
  を微調整できるフレームワーク [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory)
  がリリースされました。複数の解像度での微調整に対応しています。ぜひご利用ください！- 🔥**ニュース**: ```2024/10/10```:
  技術報告書を更新し、より詳細なトレーニング情報とデモを追加しました。
- 🔥 **ニュース**: ```2024/10/10```: 技術報告書を更新しました。[こちら](https://arxiv.org/pdf/2408.06072)
  をクリックしてご覧ください。さらにトレーニングの詳細とデモを追加しました。デモを見るには[こちら](https://yzy-thu.github.io/CogVideoX-demo/)
  をクリックしてください。
- 🔥**ニュース**: ```2024/10/09```: 飛書の[技術ドキュメント](https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh)
  でCogVideoXの微調整ガイドを公開しています。分配の自由度をさらに高めるため、公開されているドキュメント内のすべての例が完全に再現可能です。
- 🔥**ニュース**: ```2024/9/19```: CogVideoXシリーズの画像生成ビデオモデル **CogVideoX-5B-I2V**
  をオープンソース化しました。このモデルは、画像を背景入力として使用し、プロンプトワードと組み合わせてビデオを生成することができ、より高い制御性を提供します。これにより、CogVideoXシリーズのモデルは、テキストからビデオ生成、ビデオの継続、画像からビデオ生成の3つのタスクをサポートするようになりました。オンラインでの[体験](https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space)
  をお楽しみください。
- 🔥🔥 **ニュース**: ```2024/9/19```:
  CogVideoXのトレーニングプロセスでビデオデータをテキスト記述に変換するために使用されるキャプションモデル [CogVLM2-Caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption)
  をオープンソース化しました。ダウンロードしてご利用ください。
- 🔥 ```2024/8/27```: CogVideoXシリーズのより大きなモデル **CogVideoX-5B**
  をオープンソース化しました。モデルの推論性能を大幅に最適化し、推論のハードルを大幅に下げました。`GTX 1080TI` などの旧型GPUで
  **CogVideoX-2B** を、`RTX 3060` などのデスクトップGPUで **CogVideoX-5B**
  モデルを実行できます。依存関係を更新・インストールするために、[要件](requirements.txt)
  を厳守し、推論コードは [cli_demo](inference/cli_demo.py) を参照してください。さらに、**CogVideoX-2B** モデルのオープンソースライセンスが
  **Apache 2.0 ライセンス** に変更されました。
- 🔥 ```2024/8/6```: **CogVideoX-2B** 用の **3D Causal VAE** をオープンソース化しました。これにより、ビデオをほぼ無損失で再構築することができます。
- 🔥 ```2024/8/6```: CogVideoXシリーズのビデオ生成モデルの最初のモデル、**CogVideoX-2B** をオープンソース化しました。
- 🌱 **ソース**: ```2022/5/19```: CogVideoビデオ生成モデルをオープンソース化しました（現在、`CogVideo`
  ブランチで確認できます）。これは、トランスフォーマーに基づく初のオープンソース大規模テキスト生成ビデオモデルです。技術的な詳細については、[ICLR'23論文](https://arxiv.org/abs/2205.15868)
  をご覧ください。

**より強力なモデルが、より大きなパラメータサイズで登場予定です。お楽しみに！**

## 目次

特定のセクションにジャンプ：

- [クイックスタート](#クイックスタート)
    - [SAT](#sat)
    - [Diffusers](#Diffusers)
- [CogVideoX-2B ギャラリー](#CogVideoX-2B-ギャラリー)
- [モデル紹介](#モデル紹介)
- [プロジェクト構造](#プロジェクト構造)
    - [推論](#推論)
    - [sat](#sat)
    - [ツール](#ツール)
- [プロジェクト計画](#プロジェクト計画)
- [モデルライセンス](#モデルライセンス)
- [CogVideo(ICLR'23)モデル紹介](#CogVideoICLR23)
- [引用](#引用)

## クイックスタート

### プロンプトの最適化

モデルを実行する前に、[こちら](inference/convert_demo.py)
を参考にして、GLM-4（または同等の製品、例えばGPT-4）の大規模モデルを使用してどのようにモデルを最適化するかをご確認ください。これは非常に重要です。モデルは長いプロンプトでトレーニングされているため、良いプロンプトがビデオ生成の品質に直接影響を与えます。

### SAT

[sat_demo](sat/README.md) の指示に従ってください:
SATウェイトの推論コードと微調整コードが含まれています。CogVideoXモデル構造に基づいて改善することをお勧めします。革新的な研究者は、このコードを使用して迅速なスタッキングと開発を行うことができます。

### Diffusers

```
pip install -r requirements.txt
```

次に [diffusers_demo](inference/cli_demo.py) を参照してください: 推論コードの詳細な説明が含まれており、一般的なパラメータの意味についても言及しています。

量子化推論の詳細については、[diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao/) を参照してください。Diffusers
と TorchAO を使用することで、量子化推論も可能となり、メモリ効率の良い推論や、コンパイル時に場合によっては速度の向上が期待できます。A100
および H100
上でのさまざまな設定におけるメモリおよび時間のベンチマークの完全なリストは、[diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao)
に公開されています。

## Gallery

### CogVideoX-5B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/cf5953ea-96d3-48fd-9907-c4708752c714" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fe0a78e6-b669-4800-8cf0-b5f9b5145b52" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/c182f606-8f8c-421d-b414-8487070fcfcb" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7db2bbce-194d-434d-a605-350254b6c298" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/62b01046-8cab-44cc-bd45-4d965bb615ec" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d78e552a-4b3f-4b81-ac3f-3898079554f6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/30894f12-c741-44a2-9e6e-ddcacc231e5b" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/926575ca-7150-435b-a0ff-4900a963297b" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-2B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9de41efd-d4d1-4095-aeda-246dd834e91d" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/941d6661-6a8d-4a1b-b912-59606f0b2841" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/938529c4-91ae-4f60-b96b-3c3947fa63cb" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

ギャラリーの対応するプロンプトワードを表示するには、[こちら](resources/galary_prompt.md)をクリックしてください

## モデル紹介

CogVideoXは、[清影](https://chatglm.cn/video?fr=osm_cogvideox) と同源のオープンソース版ビデオ生成モデルです。
以下の表に、提供しているビデオ生成モデルの基本情報を示します:

<table  style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">モデル名</th>
    <th style="text-align: center;">CogVideoX-2B</th>
    <th style="text-align: center;">CogVideoX-5B</th>
    <th style="text-align: center;">CogVideoX-5B-I2V </th>
  </tr>
  <tr>
    <td style="text-align: center;">推論精度</td>
    <td style="text-align: center;"><b>FP16*(推奨)</b>, BF16, FP32, FP8*, INT8, INT4は非対応</td>
    <td colspan="2" style="text-align: center;"><b>BF16(推奨)</b>, FP16, FP32, FP8*, INT8, INT4は非対応</td>
  </tr>
  <tr>
    <td style="text-align: center;">単一GPUのメモリ消費<br></td>
    <td style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> FP16: 18GB <br><b>diffusers FP16: 4GBから* </b><br><b>diffusers INT8(torchao): 3.6GBから*</b></td>
    <td colspan="2" style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> BF16: 26GB <br><b>diffusers BF16 : 5GBから* </b><br><b>diffusers INT8(torchao): 4.4GBから* </b></td>
  </tr>
  <tr>
    <td style="text-align: center;">マルチGPUのメモリ消費</td>
    <td style="text-align: center;"><b>FP16: 10GB* using diffusers</b><br></td>
    <td colspan="2" style="text-align: center;"><b>BF16: 15GB* using diffusers</b><br></td>
  </tr>
  <tr>
    <td style="text-align: center;">推論速度<br>(ステップ = 50, FP/BF16)</td>
    <td style="text-align: center;">単一A100: 約90秒<br>単一H100: 約45秒</td>
    <td colspan="2" style="text-align: center;">単一A100: 約180秒<br>単一H100: 約90秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">ファインチューニング精度</td>
    <td style="text-align: center;"><b>FP16</b></td>
    <td colspan="2" style="text-align: center;"><b>BF16</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">ファインチューニング時のメモリ消費</td>
    <td style="text-align: center;">47 GB (bs=1, LORA)<br> 61 GB (bs=2, LORA)<br> 62GB (bs=1, SFT)</td>
    <td style="text-align: center;">63 GB (bs=1, LORA)<br> 80 GB (bs=2, LORA)<br> 75GB (bs=1, SFT)<br></td>
    <td style="text-align: center;">78 GB (bs=1, LORA)<br> 75GB (bs=1, SFT, 16GPU)<br></td>
  </tr>
  <tr>
    <td style="text-align: center;">プロンプト言語</td>
    <td colspan="3" style="text-align: center;">英語*</td>
  </tr>
  <tr>
    <td style="text-align: center;">プロンプトの最大トークン数</td>
    <td colspan="3" style="text-align: center;">226トークン</td>
  </tr>
  <tr>
    <td style="text-align: center;">ビデオの長さ</td>
    <td colspan="3" style="text-align: center;">6秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">フレームレート</td>
    <td colspan="3" style="text-align: center;">8フレーム/秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">ビデオ解像度</td>
    <td colspan="3" style="text-align: center;">720 * 480、他の解像度は非対応(ファインチューニング含む)</td>
  </tr>
  <tr>
    <td style="text-align: center;">位置エンコーディング</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_rope_pos_embed + learnable_pos_embed</td>
  </tr>
  <tr>
    <td style="text-align: center;">ダウンロードリンク (Diffusers)</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-2b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-2b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b-I2V">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b-I2V">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b-I2V">🟣 WiseModel</a></td>
  </tr>
  <tr>
    <td style="text-align: center;">ダウンロードリンク (SAT)</td>
    <td colspan="3" style="text-align: center;"><a href="./sat/README_ja.md">SAT</a></td>
  </tr>
</table>

**データ解説**

+ diffusersライブラリを使用してテストする際には、`diffusers`ライブラリが提供する全ての最適化が有効になっています。この方法は
  **NVIDIA A100 / H100**以外のデバイスでのメモリ/メモリ消費のテストは行っていません。通常、この方法は**NVIDIA
  Ampereアーキテクチャ**
  以上の全てのデバイスに適応できます。最適化を無効にすると、メモリ消費は倍増し、ピークメモリ使用量は表の3倍になりますが、速度は約3〜4倍向上します。以下の最適化を部分的に無効にすることが可能です:

```
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

+ マルチGPUで推論する場合、`enable_sequential_cpu_offload()`最適化を無効にする必要があります。
+ INT8モデルを使用すると推論速度が低下しますが、これはメモリの少ないGPUで正常に推論を行い、ビデオ品質の損失を最小限に抑えるための措置です。推論速度は大幅に低下します。
+ CogVideoX-2Bモデルは`FP16`精度でトレーニングされており、CogVideoX-5Bモデルは`BF16`
  精度でトレーニングされています。推論時にはモデルがトレーニングされた精度を使用することをお勧めします。
+ [PytorchAO](https://github.com/pytorch/ao)および[Optimum-quanto](https://github.com/huggingface/optimum-quanto/)
  は、CogVideoXのメモリ要件を削減するためにテキストエンコーダ、トランスフォーマ、およびVAEモジュールを量子化するために使用できます。これにより、無料のT4
  Colabやより少ないメモリのGPUでモデルを実行することが可能になります。同様に重要なのは、TorchAOの量子化は`torch.compile`
  と完全に互換性があり、推論速度を大幅に向上させることができる点です。`NVIDIA H100`およびそれ以上のデバイスでは`FP8`
  精度を使用する必要があります。これには、`torch`、`torchao`、`diffusers`、`accelerate`
  Pythonパッケージのソースコードからのインストールが必要です。`CUDA 12.4`の使用をお勧めします。
+ 推論速度テストも同様に、上記のメモリ最適化方法を使用しています。メモリ最適化を使用しない場合、推論速度は約10％向上します。
  `diffusers`バージョンのモデルのみが量子化をサポートしています。
+ モデルは英語入力のみをサポートしており、他の言語は大規模モデルの改善を通じて英語に翻訳できます。
+ モデルのファインチューニングに使用されるメモリは`8 * H100`環境でテストされています。プログラムは自動的に`Zero 2`
  最適化を使用しています。表に具体的なGPU数が記載されている場合、ファインチューニングにはその数以上のGPUが必要です。

## 友好的リンク

コミュニティからの貢献を大歓迎し、私たちもオープンソースコミュニティに積極的に貢献しています。以下の作品はすでにCogVideoXに対応しており、ぜひご利用ください：

+ [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun):
  CogVideoX-Funは、CogVideoXアーキテクチャを基にした改良パイプラインで、自由な解像度と複数の起動方法をサポートしています。
+ [CogStudio](https://github.com/pinokiofactory/cogstudio): CogVideo の Gradio Web UI の別のリポジトリ。より高機能な Web
  UI をサポートします。
+ [Xorbits Inference](https://github.com/xorbitsai/inference):
  強力で包括的な分散推論フレームワークであり、ワンクリックで独自のモデルや最新のオープンソースモデルを簡単にデプロイできます。
+ [ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper)
  ComfyUIフレームワークを使用して、CogVideoXをワークフローに統合します。
+ [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys): VideoSysは、使いやすく高性能なビデオ生成インフラを提供し、最新のモデルや技術を継続的に統合しています。
+ [AutoDLイメージ](https://www.codewithgpu.com/i/THUDM/CogVideo/CogVideoX-5b-demo): コミュニティメンバーが提供するHuggingface
  Spaceイメージのワンクリックデプロイメント。
+ [インテリアデザイン微調整モデル](https://huggingface.co/collections/bertjiazheng/koolcogvideox-66e4762f53287b7f39f8f3ba):
  は、CogVideoXを基盤にした微調整モデルで、インテリアデザイン専用に設計されています。
+ [xDiT](https://github.com/xdit-project/xDiT):
  xDiTは、複数のGPUクラスター上でDiTsを並列推論するためのエンジンです。xDiTはリアルタイムの画像およびビデオ生成サービスをサポートしています。
+ [CogVideoX-Interpolation](https://github.com/feizc/CogvideX-Interpolation):
  キーフレーム補間生成において、より大きな柔軟性を提供することを目的とした、CogVideoX構造を基にした修正版のパイプライン。
+ [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): DiffSynth
  Studioは、拡散エンジンです。テキストエンコーダー、UNet、VAEなどを含むアーキテクチャを再構築し、オープンソースコミュニティモデルとの互換性を維持しつつ、計算性能を向上させました。このフレームワークはCogVideoXに適応しています。

## プロジェクト構造

このオープンソースリポジトリは、**CogVideoX** オープンソースモデルの基本的な使用方法と微調整の例を迅速に開始するためのガイドです。

### Colabでのクイックスタート

無料のColab T4上で直接実行できる3つのプロジェクトを提供しています。

+ [CogVideoX-5B-T2V-Colab.ipynb](https://colab.research.google.com/drive/1pCe5s0bC_xuXbBlpvIH1z0kfdTLQPzCS?usp=sharing):
  CogVideoX-5B テキストからビデオへの生成用Colabコード。
+ [CogVideoX-5B-T2V-Int8-Colab.ipynb](https://colab.research.google.com/drive/1DUffhcjrU-uz7_cpuJO3E_D4BaJT7OPa?usp=sharing):
  CogVideoX-5B テキストからビデオへの量子化推論用Colabコード。1回の実行に約30分かかります。
+ [CogVideoX-5B-I2V-Colab.ipynb](https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX?usp=sharing):
  CogVideoX-5B 画像からビデオへの生成用Colabコード。
+ [CogVideoX-5B-V2V-Colab.ipynb](https://colab.research.google.com/drive/1comfGAUJnChl5NwPuO8Ox5_6WCy4kbNN?usp=sharing):
  CogVideoX-5B ビデオからビデオへの生成用Colabコード。

### Inference

+ [cli_demo](inference/cli_demo.py): 推論コードの詳細な説明が含まれており、一般的なパラメータの意味についても言及しています。
+ [cli_demo_quantization](inference/cli_demo_quantization.py):
  量子化モデル推論コードで、低メモリのデバイスでも実行可能です。また、このコードを変更して、FP8 精度の CogVideoX
  モデルの実行をサポートすることもできます。
+ [diffusers_vae_demo](inference/cli_vae_demo.py): VAE推論コードの実行には現在71GBのメモリが必要ですが、将来的には最適化される予定です。
+ [space demo](inference/gradio_composite_demo): Huggingface Spaceと同じGUIコードで、フレーム補間や超解像ツールが組み込まれています。

<div style="text-align: center;">
    <img src="resources/web_demo.png" style="width: 100%; height: auto;" />
</div>

+ [convert_demo](inference/convert_demo.py):
  ユーザー入力をCogVideoXに適した形式に変換する方法。CogVideoXは長いキャプションでトレーニングされているため、入力テキストをLLMを使用してトレーニング分布と一致させる必要があります。デフォルトではGLM-4を使用しますが、GPT、Geminiなどの他のLLMに置き換えることもできます。
+ [gradio_web_demo](inference/gradio_web_demo.py): CogVideoX-2B / 5B モデルを使用して動画を生成する方法を示す、シンプルな
  Gradio Web UI デモです。私たちの Huggingface Space と同様に、このスクリプトを使用して Web デモを起動することができます。

### finetune

+ [train_cogvideox_lora](finetune/README_ja.md): CogVideoX diffusers 微調整方法の詳細な説明が含まれています。このコードを使用して、自分のデータセットで
  CogVideoX を微調整することができます。

### sat

+ [sat_demo](sat/README.md):
  SATウェイトの推論コードと微調整コードが含まれています。CogVideoXモデル構造に基づいて改善することをお勧めします。革新的な研究者は、このコードを使用して迅速なスタッキングと開発を行うことができます。

### ツール

このフォルダには、モデル変換/キャプション生成などのツールが含まれています。

+ [convert_weight_sat2hf](tools/convert_weight_sat2hf.py): SAT モデルの重みを Huggingface モデルの重みに変換します。
+ [caption_demo](tools/caption/README_ja.md): Caption ツール、ビデオを理解してテキストで出力するモデル。
+ [export_sat_lora_weight](tools/export_sat_lora_weight.py): SAT ファインチューニングモデルのエクスポートツール、SAT Lora
  Adapter を diffusers 形式でエクスポートします。
+ [load_cogvideox_lora](tools/load_cogvideox_lora.py): diffusers 版のファインチューニングされた Lora Adapter
  をロードするためのツールコード。
+ [llm_flux_cogvideox](tools/llm_flux_cogvideox/llm_flux_cogvideox.py): オープンソースのローカル大規模言語モデル +
  Flux + CogVideoX を使用して自動的に動画を生成します。
+ [parallel_inference_xdit](tools/parallel_inference/parallel_inference_xdit.py)：
  [xDiT](https://github.com/xdit-project/xDiT)
  によってサポートされ、ビデオ生成プロセスを複数の GPU で並列化します。
+ [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory): CogVideoXの低コスト微調整フレームワークで、
  `diffusers`バージョンのモデルに適応しています。より多くの解像度に対応し、単一の4090 GPUでCogVideoX-5Bの微調整が可能です。

## CogVideo(ICLR'23)

論文の公式リポジトリ: [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)
は [CogVideo branch](https://github.com/THUDM/CogVideo/tree/CogVideo) にあります。

**CogVideoは比較的高フレームレートのビデオを生成することができます。**
32フレームの4秒間のクリップが以下に示されています。

![High-frame-rate sample](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/appendix-sample-highframerate.png)

![Intro images](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/intro-image.png)
<div align="center">
  <video src="https://github.com/user-attachments/assets/2fa19651-e925-4a2a-b8d6-b3f216d490ba" width="80%" controls autoplay></video>
</div>


CogVideoのデモは [https://models.aminer.cn/cogvideo](https://models.aminer.cn/cogvideo/) で体験できます。
*元の入力は中国語です。*

## 引用

🌟 私たちの仕事が役立つと思われた場合、ぜひスターを付けていただき、論文を引用してください。

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

あなたの貢献をお待ちしています！詳細は[こちら](resources/contribute_ja.md)をクリックしてください。

## ライセンス契約

このリポジトリのコードは [Apache 2.0 License](LICENSE) の下で公開されています。

CogVideoX-2B モデル (対応するTransformersモジュールやVAEモジュールを含む) は
[Apache 2.0 License](LICENSE) の下で公開されています。

CogVideoX-5B モデル（Transformers モジュール、画像生成ビデオとテキスト生成ビデオのバージョンを含む） は
[CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) の下で公開されています。
