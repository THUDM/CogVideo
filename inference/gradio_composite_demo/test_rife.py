import utils

import torch

from inference.gradio_composite_demo.rife_model import load_rife_model, rife_inference_with_path

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    model_path = "/media/gpt4-pdf-chatbot-langchain/ECCV2022-RIFE/train_log"
    # video_path = "/media/gpt4-pdf-chatbot-langchain/CogVideo/inference/output/20240823_110325.mp4"
    model = load_rife_model(model_path)

    # rife_inference_with_path(model, video_path)
