import utils

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_sd_upscale(ckpt):
    from spandrel import ModelLoader, ImageModelDescriptor  # Simulate a step in loading

    pbar = utils.ProgressBar(1, desc="Loading upscale model")
    sd = utils.load_torch_file(ckpt, device=device)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = utils.state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).half()

    pbar.update(1)  # Update progress by 1
    return out


def test_load_sd_upscale():
    model = load_sd_upscale("/media/gpt4-pdf-chatbot-langchain/ComfyUI/models/upscale_models/RealESRNet_x4plus.pth")

    print(model.dtype)


if __name__ == "__main__":
    test_load_sd_upscale()
