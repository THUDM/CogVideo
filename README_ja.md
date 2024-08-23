# CogVideo && CogVideoX

[Read this in English.](./README_zh)

[中文阅读](./README_zh.md)

<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
🤗 <a href="https://huggingface.co/spaces/THUDM/CogVideoX" target="_blank">CogVideoX Huggingface Space</a> で体験
</p>
<p align="center">
📚 <a href="https://arxiv.org/abs/2408.06072" target="_blank">論文</a> をチェック
</p>
<p align="center">
    👋 <a href="resources/WECHAT.md" target="_blank">WeChat</a> と <a href="https://discord.gg/B94UfuhN" target="_blank">Discord</a> に参加
</p>
<p align="center">
📍 <a href="https://chatglm.cn/video?fr=osm_cogvideox">清影</a> と <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">APIプラットフォーム</a> を訪問して、より大規模な商用ビデオ生成モデルを体験
</p>

## 更新とニュース

- 🔥🔥 **ニュース**: ```2024/8/20```: [VEnhancer](https://github.com/Vchitect/VEnhancer) は CogVideoX
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
(推論には18GB、lora微調整には40GBが必要です)

### Diffusers

```
pip install -r requirements.txt
```

次に [diffusers_demo](inference/cli_demo.py) を参照してください: 推論コードの詳細な説明が含まれており、一般的なパラメータの意味についても言及しています。
(推論には24GBが必要で、微調整コードは開発中です)

## CogVideoX-2B ギャラリー

<div align="center">
  <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="80%" controls autoplay></video>
  <p>詳細に彫刻されたマストと帆を持つ木製の玩具船が、海の波を模倣した豪華な青いカーペットの上を滑らかに進んでいます。船体は濃い茶色に塗られ、小さな窓が付いています。カーペットは柔らかく、テクスチャーがあり、海洋の広がりを連想させる完璧な背景を提供します。船の周りにはさまざまな他の玩具や子供のアイテムがあり、遊び心のある環境を示唆しています。このシーンは、子供時代の無邪気さと想像力を捉えており、玩具船の旅は室内の幻想的な設定での無限の冒険を象徴しています。</p>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/9de41efd-d4d1-4095-aeda-246dd834e91d" width="80%" controls autoplay></video>
  <p>カメラは、黒いルーフラックを備えた白いビンテージSUVの後ろを追いかけ、急な山道をスピードアップして進みます。タイヤからほこりが舞い上がり、日光がSUVに当たり、暖かい輝きを放ちます。山道は緩やかに曲がり、他の車両は見当たりません。道の両側には赤杉の木が立ち並び、緑のパッチが点在しています。車は後ろから見て、険しい地形を楽々と進んでいるように見えます。山道自体は急な丘と山に囲まれ、上空には青い空と薄い雲が広がっています。</p>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/941d6661-6a8d-4a1b-b912-59606f0b2841" width="80%" controls autoplay></video>
  <p>色とりどりのバンダナを巻いた、擦り切れたデニムジャケットを着たストリートアーティストが、広大なコンクリートの壁の前に立ち、スプレーペイント缶を持ち、斑点のある壁にカラフルな鳥をスプレーペイントしています。</p>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/938529c4-91ae-4f60-b96b-3c3947fa63cb" width="80%" controls autoplay></video>
  <p>戦争で荒廃した都市の背景に、廃墟と崩れた壁が破壊の物語を語る中、若い少女の感動的なクローズアップがフレームに収められています。彼女の顔は灰で汚れており、周囲の混乱を静かに物語っています。彼女の目は悲しみと回復力の混じった輝きを放ち、紛争の荒廃によって無垢を失った世界の生の感情を捉えています。</p>
</div>

## モデル紹介

CogVideoXは、[清影](https://chatglm.cn/video?fr=osm_cogvideox) と同源のオープンソース版ビデオ生成モデルです。

以下の表は、現在提供しているビデオ生成モデルのリストと関連する基本情報を示しています:

| モデル名                         | CogVideoX-2B                                                                                                                                                                                        | 
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| プロンプト言語                      | 英語                                                                                                                                                                                                  | 
| 単一GPU推論 (FP16)               | 18GB using [SAT](https://github.com/THUDM/SwissArmyTransformer)   <br>  23.9GB using diffusers                                                                                                      | 
| 複数GPU推論 (FP16)               | 20GB minimum per GPU using diffusers                                                                                                                                                                |
| 微調整に必要なGPUメモリ(bs=1)          | 40GB                                                                                                                                                                                                |
| プロンプトの最大長                    | 226 トークン                                                                                                                                                                                            |
| ビデオの長さ                       | 6秒                                                                                                                                                                                                  | 
| フレームレート                      | 8フレーム                                                                                                                                                                                               | 
| 解像度                          | 720 * 480                                                                                                                                                                                           |
| 量子化推論                        | サポートされていません                                                                                                                                                                                         |          
| ダウンロードリンク (HF diffusers モデル) | 🤗 [Huggingface](https://huggingface.co/THUDM/CogVideoX-2B)   [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b)   [💫 WiseModel](https://wisemodel.cn/models/ZhipuAI/CogVideoX-2b) |
| ダウンロードリンク (SAT モデル)          | [SAT](./sat/README.md)                                                                                                                                                                              |

## 友好的リンク

コミュニティからの貢献を大歓迎し、私たちもオープンソースコミュニティに積極的に貢献しています。以下の作品はすでにCogVideoXに対応しており、ぜひご利用ください：

+ [Xorbits Inference](https://github.com/xorbitsai/inference):
  強力で包括的な分散推論フレームワークであり、ワンクリックで独自のモデルや最新のオープンソースモデルを簡単にデプロイできます。
+ [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys): VideoSysは、使いやすく高性能なビデオ生成インフラを提供し、最新のモデルや技術を継続的に統合しています。


## プロジェクト構造

このオープンソースリポジトリは、**CogVideoX** オープンソースモデルの基本的な使用方法と微調整の例を迅速に開始するためのガイドです。

### 推論

+ [diffusers_demo](inference/cli_demo.py): 推論コードの詳細な説明が含まれており、一般的なパラメータの意味についても言及しています。
+ [diffusers_vae_demo](inference/cli_vae_demo.py): VAE推論コードの実行には現在71GBのメモリが必要ですが、将来的には最適化される予定です。
+ [convert_demo](inference/convert_demo.py):
  ユーザー入力をCogVideoXに適した形式に変換する方法。CogVideoXは長いキャプションでトレーニングされているため、入力テキストをLLMを使用してトレーニング分布と一致させる必要があります。デフォルトではGLM4を使用しますが、GPT、Geminiなどの他のLLMに置き換えることもできます。
+ [gradio_web_demo](inference/gradio_web_demo.py): CogVideoX-2B モデルを使用して動画を生成する方法を示す、シンプルな
  Gradio Web UI デモです。私たちの Huggingface Space と同様に、このスクリプトを使用して Web デモを起動することができます。

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

+ [streamlit_web_demo](inference/streamlit_web_demo.py): CogVideoX-2Bモデルを使用してビデオを生成する方法を示すシンプルなstreamlit
  Webアプリケーション。

<div style="text-align: center;">
    <img src="resources/web_demo.png" style="width: 100%; height: auto;" />
</div>

### sat

+ [sat_demo](sat/README.md):
  SATウェイトの推論コードと微調整コードが含まれています。CogVideoXモデル構造に基づいて改善することをお勧めします。革新的な研究者は、このコードを使用して迅速なスタッキングと開発を行うことができます。

### ツール

このフォルダには、モデル変換/キャプション生成などのツールが含まれています。

+ [convert_weight_sat2hf](tools/convert_weight_sat2hf.py): SATモデルのウェイトをHuggingfaceモデルのウェイトに変換します。
+ [caption_demo](tools/caption): キャプションツール、ビデオを理解し、テキストで出力するモデル。

## プロジェクト計画

- [x] CogVideoXモデルのオープンソース化
    - [x] CogVideoXで使用される3D Causal VAEのオープンソース化
    - [x] CogVideoXモデルの推論例 (CLI / Webデモ)
    - [x] CogVideoXオンライン体験デモ (Huggingface Space)
    - [x] CogVideoXオープンソースモデルAPIインターフェースの例 (Huggingface)
    - [x] CogVideoXモデルの微調整例 (SAT)
    - [ ] CogVideoXモデルの微調整例 (Huggingface / SAT)
    - [ ] CogVideoX-Proのオープンソース化 (CogVideoX-2Bスイートに適応)
    - [x] CogVideoX技術レポートの公開

私たちはあなたの貢献を歓迎します。詳細については[こちら](resources/contribute.md)をクリックしてください。

## モデルライセンス

このリポジトリのコードは [Apache 2.0 ライセンス](LICENSE) の下で公開されています。

モデルのウェイトと実装コードは [CogVideoX LICENSE](MODEL_LICENSE) の下で公開されています。

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

- [x] CogVideoX モデルのオープンソース化
    - [x] CogVideoX モデル推論サンプル (CLI / Web デモ)
    - [x] CogVideoX オンライン体験サンプル (Huggingface Space)
    - [x] CogVideoX オープンソースAPIインターフェースサンプル (Huggingface)
    - [x] CogVideoX モデルの微調整サンプル (SAT)
    - [ ] CogVideoX モデルの微調整サンプル (Huggingface / SAT)
    - [ ] CogVideoX-Pro オープンソース化 (CogVideoX-2B スイートに対応)
    - [X] CogVideoX 技術レポート公開

私たちは皆さんの貢献を歓迎しています。詳しくは[こちら](resources/contribute_zh.md)をご覧ください。

## モデルライセンス

本リポジトリのコードは [Apache 2.0 ライセンス](LICENSE) の下で公開されています。

本モデルのウェイトと実装コードは [CogVideoX LICENSE](MODEL_LICENSE) ライセンスに基づいて公開されています。