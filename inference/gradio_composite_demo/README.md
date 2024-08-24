
#### Gradio Composite Demo

这里集成了CogVideo模型的Gradio Demo，可以直接在浏览器中进行视频的推理。
支持UpScale, RIFE等功能

#### 使用方法


1. 下载CogVideo模型
2. [RIFE模型](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing) (百度网盘链接:https://pan.baidu.com/share/init?surl=u6Q7-i4Hu4Vx9_5BJibPPA 密码:hfk3，把压缩包解开后放在 train_log/*)
3. UP_SCALE模型，可以使用[RealESRNet模型](https://huggingface.co/ai-forever/Real-ESRGAN)，也可以使用其他的UP_SCALE模型，只需要修改代码中的UP_SCALE_MODEL_CKPT路径即可


#### 环境配置
 

MODEL_PATH=CogVideo模型路径;
OPENAI_API_KEY= your_api_key;
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4;
RIFE_MODEL_PATH= RIFE模型路径 train_log;
UP_SCALE_MODEL_CKPT=UP_SCALE模型 model路径;


#### 安装依赖

```bash
pip install -r requirements.txt 
```

#### 运行

```bash
python gradio_web_demo.py
```