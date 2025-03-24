"""
This script is designed to demonstrate how to use the CogVideoX-2b VAE model for video encoding and decoding.
It allows you to encode a video into a latent representation, decode it back into a video, or perform both operations sequentially.
Before running the script, make sure to clone the CogVideoX Hugging Face model repository and set the
`{your local diffusers path}` argument to the path of the cloned repository.

Command 1: Encoding Video
Encodes the video located at ../resources/videos/1.mp4 using the CogVideoX-5b VAE model.
Memory Usage: ~18GB of GPU memory for encoding.

If you do not have enough GPU memory, we provide a pre-encoded tensor file (encoded.pt) in the resources folder,
and you can still run the decoding command.

$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode encode

Command 2: Decoding Video

Decodes the latent representation stored in encoded.pt back into a video.
Memory Usage: ~4GB of GPU memory for decoding.
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --encoded_path ./encoded.pt --mode decode

Command 3: Encoding and Decoding Video
Encodes the video located at ../resources/videos/1.mp4 and then immediately decodes it.
Memory Usage: 34GB for encoding + 19GB for decoding (sequentially).
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode both
"""

import argparse
import torch
import imageio
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms
import numpy as np


def encode_video(model_path, video_path, dtype, device):
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

    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)

    model.enable_slicing()
    model.enable_tiling()

    video_reader = imageio.get_reader(video_path, "ffmpeg")

    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    video_reader.close()

    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)

    with torch.no_grad():
        encoded_frames = model.encode(frames_tensor)[0].sample()
    return encoded_frames


def decode_video(model_path, encoded_tensor_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and decodes the encoded video frames.

    Parameters:
    - model_path (str): The path to the pre-trained model.
    - encoded_tensor_path (str): The path to the encoded tensor file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The decoded video frames.
    """
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)
    with torch.no_grad():
        decoded_frames = model.decode(encoded_frames).sample
    return decoded_frames


def save_video(tensor, output_path):
    """
    Saves the video frames to a video file.

    Parameters:
    - tensor (torch.Tensor): The video frames' tensor.
    - output_path (str): The path to save the output video.
    """
    tensor = tensor.to(dtype=torch.float32)
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)
    writer = imageio.get_writer(output_path + "/output.mp4", fps=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVideoX encode/decode demo")
    parser.add_argument(
        "--model_path", type=str, required=True, help="The path to the CogVideoX model"
    )
    parser.add_argument("--video_path", type=str, help="The path to the video file (for encoding)")
    parser.add_argument(
        "--encoded_path", type=str, help="The path to the encoded tensor file (for decoding)"
    )
    parser.add_argument(
        "--output_path", type=str, default=".", help="The path to save the output file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["encode", "decode", "both"],
        required=True,
        help="Mode: encode, decode, or both",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="The data type for computation (e.g., 'float16' or 'bfloat16')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for computation (e.g., 'cuda' or 'cpu')",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if args.mode == "encode":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        print(
            f"Finished encoding the video to a tensor, save it to a file at {encoded_output}/encoded.pt"
        )
    elif args.mode == "decode":
        assert args.encoded_path, "Encoded tensor path must be provided for decoding."
        decoded_output = decode_video(args.model_path, args.encoded_path, dtype, device)
        save_video(decoded_output, args.output_path)
        print(
            f"Finished decoding the video and saved it to a file at {args.output_path}/output.mp4"
        )
    elif args.mode == "both":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        decoded_output = decode_video(
            args.model_path, args.output_path + "/encoded.pt", dtype, device
        )
        save_video(decoded_output, args.output_path)
