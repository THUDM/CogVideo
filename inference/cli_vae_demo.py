"""
This script demonstrates how to encode video frames using a pre-trained CogVideoX model with 🤗 Huggingface Diffusers.

Note:
    This script requires the `diffusers>=0.30.0` library to be installed.
    If the video appears “completely green” and cannot be viewed, please switch to a different player to watch it. This is a normal phenomenon.
    Cost 71GB of GPU memory for encoding a 1-minute video at 720p resolution.

Run the script:
    $ python cli_demo.py --model_path THUDM/CogVideoX-2b --video_path path/to/video.mp4 --output_path path/to/output

"""

import argparse
import torch
import imageio
import numpy as np
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms


def vae_demo(model_path, video_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and encodes the video frames.

    Parameters:
    - model_path (str): The path to the pre-trained model.
    - video_path (str): The path to the video file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The encoded video frames.
    """
    # Load the pre-trained model
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)

    # Load video frames
    video_reader = imageio.get_reader(video_path, 'ffmpeg')
    frames = []
    for frame in video_reader:
        frames.append(frame)
    video_reader.close()

    # Transform frames to Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    frames_tensor = torch.stack([transform(frame) for frame in frames]).to(device)

    # Add batch dimension and reshape to [1, 3, 49, 480, 720]
    frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(dtype).to(device)

    # Run the model with Encoder and Decoder
    with torch.no_grad():
        output = model(frames_tensor)

    return output


def save_video(tensor, output_path):
    """
    Saves the encoded video frames to a video file.

    Parameters:
    - tensor (torch.Tensor): The encoded video frames.
    - output_path (str): The path to save the output video.
    """
    # Remove batch dimension and permute back to [49, 480, 720, 3]
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()

    # Clip values to [0, 1] and convert to uint8
    frames = np.clip(frames, 0, 1)
    frames = (frames * 255).astype(np.uint8)

    # Save frames to video
    writer = imageio.get_writer(output_path + "/output.mp4", fps=30)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CogVideoX model to Diffusers")
    parser.add_argument("--model_path", type=str, required=True, help="The path to the CogVideoX model")
    parser.add_argument("--video_path", type=str, required=True, help="The path to the video file")
    parser.add_argument(
        "--output_path", type=str, default="./", help="The path to save the output video"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", help="The data type for computation (e.g., 'float16' or 'float32')"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to use for computation (e.g., 'cuda' or 'cpu')"
    )
    args = parser.parse_args()

    # Set device and dtype
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    output = vae_demo(args.model_path, args.video_path, dtype, device)
    save_video(output, args.output_path)