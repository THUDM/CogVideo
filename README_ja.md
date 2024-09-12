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
    👋 <a href="resources/WECHAT.md" target="_blank">WeChat</a> と <a href="https://discord.gg/Qqjtj69f" target="_blank">Discord</a> に参加
</p>
<p align="center">
📍 <a href="https://chatglm.cn/video?lang=en?fr=osm_cogvideo">清影</a> と <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">APIプラットフォーム</a> を訪問して、より大規模な商用ビデオ生成モデルを体験
</p>

## 更新とニュース

- 🔥🔥 **ニュース**: ```2024/8/29```: `pipe.enable_sequential_cpu_offload()` と `pipe.vae.enable_slicing()`
  をCogVideoX-5Bの推論コードに追加することで、VRAM使用量を`5GB`
  まで削減できます。更新された[cli_demo](inference/cli_demo.py)をご覧ください。
- 🔥**ニュース**: ```2024/8/27```: **CogVideoX-2B** モデルのオープンソースライセンスが **Apache 2.0 ライセンス**
  に変更されました。
- 🔥**ニュース**: ```2024/8/27```: CogVideoX シリーズのより大きなモデル **CogVideoX-5B** をオープンソース化しました。
  モデルの推論性能を大幅に最適化し、推論のハードルを大幅に下げました。`GTX 1080TI` などの旧型GPUで **CogVideoX-2B**
  を、`RTX 3060` などのミドル
- 🔥**ニュース**: ```2024/8/20```: [VEnhancer](https://github.com/Vchitect/VEnhancer) は CogVideoX
  が生成したビデオの強化をサポートしました。より高い解像度とより高品質なビデオレンダリングを実現します。[チュートリアル](tools/venhancer/README_ja.md)
  に従って、ぜひお試しください。
- 🔥**ニュース**: 2024/8/15: CogVideoX の依存関係である`SwissArmyTransformer`の依存が`0.4.12`
  にアップグレードされました。これにより、微調整の際に`SwissArmyTransformer`
  をソースコードからインストールする必要がなくなりました。同時に、`Tied VAE` 技術が `diffusers`
  ライブラリの実装に適用されました。`diffusers` と `accelerate` ライブラリをソースコードからインストールしてください。CogVdideoX
  の推論には 12GB の VRAM だけが必要です。 推論コードの修正が必要です。[cli_demo](inference/cli_demo.py)をご確認ください。
- 🔥 **ニュース**: ```2024/8/12```: CogVideoX
  論文がarxivにアップロードされました。ぜひ[論文](https://arxiv.org/abs/2408.06072)をご覧ください。
- 🔥 **ニュース**: ```2024/8/7```: CogVideoX は `diffusers` バージョン 0.30.0 に統合されました。単一の 3090 GPU
  で推論を実行できます。詳細については [コード](inference/cli_demo.py) を参照してください。
- 🔥 **ニュース**: ```2024/8/6```: **CogVideoX-2B** で使用される **3D Causal VAE** もオープンソース化しました。これにより、ビデオをほぼ無損失で再構築できます。
- 🔥 **ニュース**: ```2024/8/6```: **CogVideoX-2B**、CogVideoXシリーズのビデオ生成モデルの最初のモデルをオープンソース化しました。
- 🌱 **ソース**: ```2022/5/19```: **CogVideo** (現在 `CogVideo` ブランチで確認できます)
  をオープンソース化しました。これは、最初のオープンソースの事前学習済みテキストからビデオ生成モデルであり、技術的な詳細については [ICLR'23 CogVideo 論文](https://arxiv.org/abs/2205.15868)
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

CogVideoXは[清影](https://chatglm.cn/video?lang=en?fr=osm_cogvideo) 同源のオープンソース版動画生成モデルです。
以下の表は、提供されている動画生成モデルに関する基本情報を示しています。

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">モデル名</th>
    <th style="text-align: center;">CogVideoX-2B</th>
    <th style="text-align: center;">CogVideoX-5B</th>
  </tr>
  <tr>
    <td style="text-align: center;">モデル紹介</td>
    <td style="text-align: center;">入門モデルで、互換性を重視。運用および二次開発のコストが低い。</td>
    <td style="text-align: center;">動画生成品質が高く、視覚効果がより優れた大型モデル。</td>
  </tr>
  <tr>
    <td style="text-align: center;">推論精度</td>
    <td style="text-align: center;"><b>FP16*(推奨)</b>, BF16, FP32, FP8*(E4M3, E5M2), INT8, INT4は非対応</td>
    <td style="text-align: center;"><b>BF16(推奨)</b>, FP16, FP32, FP8*(E4M3, E5M2), INT8, INT4は非対応</td>
  </tr>
  <tr>
    <td style="text-align: center;">シングルGPU VRAM消費量<br></td>
    <td style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> FP16: 18GB <br><b>diffusers FP16: 4GBから*</b><br><b>diffusers INT8(torchao): 3.6GBから*</b></td>
    <td style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> BF16: 26GB <br><b>diffusers BF16: 5GBから*</b><br><b>diffusers INT8(torchao): 4.4GBから*</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">複数GPUの推論メモリ消費量</td>
    <td style="text-align: center;"><b>FP16: 10GB* using diffusers</b></td>
    <td style="text-align: center;"><b>BF16: 15GB* using diffusers</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">推論速度<br>(Step = 50)</td>
    <td style="text-align: center;">FP16: ~90* s</td>
    <td style="text-align: center;">BF16: ~180* s</td>
  </tr>
  <tr>
    <td style="text-align: center;">微調整精度</td>
    <td style="text-align: center;"><b>FP16</b></td>
    <td style="text-align: center;"><b>BF16</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">微調整時のメモリ消費量 (1GPUあたり)</td>
    <td style="text-align: center;">47 GB (bs=1, LORA)<br> 61 GB (bs=2, LORA)<br> 62GB (bs=1, SFT)</td>
    <td style="text-align: center;">63 GB (bs=1, LORA)<br> 80 GB (bs=2, LORA)<br> 75GB (bs=1, SFT)</td>
  </tr>
  <tr>
    <td style="text-align: center;">プロンプト言語</td>
    <td colspan="2" style="text-align: center;">英語*</td>
  </tr>
  <tr>
    <td style="text-align: center;">プロンプトの長さ上限</td>
    <td colspan="2" style="text-align: center;">226トークン</td>
  </tr>
  <tr>
    <td style="text-align: center;">動画の長さ</td>
    <td colspan="2" style="text-align: center;">6秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">フレームレート</td>
    <td colspan="2" style="text-align: center;">8フレーム/秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">動画の解像度</td>
    <td colspan="2" style="text-align: center;">720 * 480、他の解像度はサポートされていません（微調整も含む）</td>
  </tr>
  <tr>
    <td style="text-align: center;">位置エンコード</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_rope_pos_embed</td>
  </tr>
  <tr>
    <td style="text-align: center;">ダウンロードリンク (Diffusers)</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-2b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-2b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b">🟣 WiseModel</a></td>
  </tr>
  <tr>
    <td style="text-align: center;">ダウンロードリンク (SAT)</td>
    <td colspan="2" style="text-align: center;"><a href="./sat/README_zh.md">SAT</a></td>
  </tr>
</table>

**データ解説**

+ `diffusers` ライブラリを使用してテストする際、`diffusers` ライブラリに付属するすべての最適化を有効にしました。このソリューションは、
  **NVIDIA A100 / H100** 以外のデバイスでの実際のVRAM/メモリ使用量についてはテストされていません。一般的に、このソリューションは
  **NVIDIA Ampereアーキテクチャ**
  以上のすべてのデバイスに適応できます。最適化を無効にすると、VRAM使用量が大幅に増加し、表の約3倍のピークVRAMを使用しますが、速度は3-4倍向上します。以下の最適化の一部を選択的に無効にすることができます:

```
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

+ マルチGPU推論を行う際には、`enable_sequential_cpu_offload()` の最適化を無効にする必要があります。
+ INT8モデルを使用すると推論速度が低下します。これは、ビデオ品質の損失を最小限に抑えながら、VRAMが少ないGPUでも正常に推論できるようにするためですが、推論速度は大幅に低下します。
+ 2Bモデルは `FP16` 精度でトレーニングされ、5Bモデルは `BF16` 精度でトレーニングされています。推論には、モデルがトレーニングされた精度を使用することをお勧めします。
+ [PytorchAO](https://github.com/pytorch/ao) および [Optimum-quanto](https://github.com/huggingface/optimum-quanto/)
  は、CogVideoXのメモリ要件を削減するためにテキストエンコーダー、トランスフォーマー、およびVAEモジュールを量子化するために使用できます。これにより、無料のT4
  ColabまたはVRAMが少ないGPUでモデルを実行できるようになります。また、TorchAO量子化は `torch.compile`
  と完全に互換性があり、推論速度を大幅に向上させることができます。`NVIDIA H100` 以上のデバイスでは `FP8`
  精度を使用する必要があり、これには `torch`、`torchao`、`diffusers`、および `accelerate` Python
  パッケージをソースからインストールする必要があります。`CUDA 12.4` が推奨されます。
+

推論速度テストでも、上記のVRAM最適化スキームを使用しました。VRAMの最適化を行わない場合、推論速度は約10％向上します。量子化をサポートするのは `diffusers`
バージョンのモデルのみです。

+ モデルは英語入力のみをサポートしており、他の言語は大規模なモデルでのリファイン時に英語に翻訳できます。

## 友好的リンク

コミュニティからの貢献を大歓迎し、私たちもオープンソースコミュニティに積極的に貢献しています。以下の作品はすでにCogVideoXに対応しており、ぜひご利用ください：

+ [Xorbits Inference](https://github.com/xorbitsai/inference):
  強力で包括的な分散推論フレームワークであり、ワンクリックで独自のモデルや最新のオープンソースモデルを簡単にデプロイできます。
+ + [ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper) ComfyUIフレームワークを使用して、CogVideoXをワークフローに統合します。
+ [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys): VideoSysは、使いやすく高性能なビデオ生成インフラを提供し、最新のモデルや技術を継続的に統合しています。
+ [AutoDLイメージ](https://www.codewithgpu.com/i/THUDM/CogVideo/CogVideoX-5b-demo): コミュニティメンバーが提供するHuggingface
  Spaceイメージのワンクリックデプロイメント。
+ [Colab Space](https://github.com/camenduru/CogVideoX-5B-jupyter) ColabでJupyter Notebookを使用してCogVideoX-5Bモデルを実行します。

## プロジェクト構造

このオープンソースリポジトリは、**CogVideoX** オープンソースモデルの基本的な使用方法と微調整の例を迅速に開始するためのガイドです。

### 推論

+ [cli_demo](inference/cli_demo.py): 推論コードの詳細な説明が含まれており、一般的なパラメータの意味についても言及しています。
+ [cli_demo_quantization](inference/cli_demo_quantization.py):
  量子化モデル推論コードで、低メモリのデバイスでも実行可能です。また、このコードを変更して、FP8 精度の CogVideoX
  モデルの実行をサポートすることもできます。
+ [diffusers_vae_demo](inference/cli_vae_demo.py): VAE推論コードの実行には現在71GBのメモリが必要ですが、将来的には最適化される予定です。
+ [space demo](inference/gradio_composite_demo): Huggingface Spaceと同じGUIコードで、フレーム補間や超解像ツールが組み込まれています。
+ [convert_demo](inference/convert_demo.py):
  ユーザー入力をCogVideoXに適した形式に変換する方法。CogVideoXは長いキャプションでトレーニングされているため、入力テキストをLLMを使用してトレーニング分布と一致させる必要があります。デフォルトではGLM-4を使用しますが、GPT、Geminiなどの他のLLMに置き換えることもできます。
+ [gradio_web_demo](inference/gradio_web_demo.py): CogVideoX-2B / 5B モデルを使用して動画を生成する方法を示す、シンプルな
  Gradio Web UI デモです。私たちの Huggingface Space と同様に、このスクリプトを使用して Web デモを起動することができます。

```shell
cd inference
# For Linux and Windows users
python gradio_web_demo.py

# For macOS with Apple Silicon users, Intel not supported, this maybe 20x slower than RTX 4090
PYTORCH_ENABLE_MPS_FALLBACK=1 python gradio_web_demo.py
```

<div style="text-align: center;">
    <img src="resources/gradio_demo.png" style="width: 100%; height: auto;" />
</div>

### sat

+ [sat_demo](sat/README.md):
  SATウェイトの推論コードと微調整コードが含まれています。CogVideoXモデル構造に基づいて改善することをお勧めします。革新的な研究者は、このコードを使用して迅速なスタッキングと開発を行うことができます。

### ツール

このフォルダには、モデル変換/キャプション生成などのツールが含まれています。

+ [convert_weight_sat2hf](tools/convert_weight_sat2hf.py): SATモデルのウェイトをHuggingfaceモデルのウェイトに変換します。
+ [caption_demo](tools/caption): キャプションツール、ビデオを理解し、テキストで出力するモデル。

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

## オープンソースプロジェクト計画

- [x] CogVideoX モデルオープンソース化
    - [x] CogVideoX モデル推論例 (CLI / Web デモ)
    - [x] CogVideoX オンライン体験例 (Huggingface Space)
    - [x] CogVideoX オープンソースモデルAPIインターフェース例 (Huggingface)
    - [x] CogVideoX モデル微調整例 (SAT)
    - [ ] CogVideoX モデル微調整例 (Huggingface Diffusers)
    - [X] CogVideoX-5B オープンソース化 (CogVideoX-2B スイートに適応)
    - [X] CogVideoX 技術報告公開
    - [X] CogVideoX 技術解説ビデオ
- [ ] CogVideoX 周辺ツール
    - [X] 基本的なビデオ超解像 / フレーム補間スイート
    - [ ] 推論フレームワーク適応
    - [ ] ComfyUI 完全エコシステムツール

あなたの貢献をお待ちしています！詳細は[こちら](resources/contribute_zh.md)をクリックしてください。

## ライセンス契約

このリポジトリのコードは [Apache 2.0 License](LICENSE) の下で公開されています。

CogVideoX-2B モデル (対応するTransformersモジュールやVAEモジュールを含む) は
[Apache 2.0 License](LICENSE) の下で公開されています。

CogVideoX-5B モデル (Transformersモジュール) は
[CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) の下で公開されています。
