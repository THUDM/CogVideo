"""
This script demonstrates how to generate a video from a text prompt using CogVideoX with 🤗Huggingface Diffusers Pipeline.

Note:
    This script requires the `diffusers>=0.30.0` library to be installed.
    If the video exported using OpenCV appears “completely green” and cannot be viewed, lease switch to a different player to watch it. This is a normal phenomenon.

Run the script:
    $ python cli_demo.py --prompt "A girl ridding a bike." --model_path THUDM/CogVideoX-2b

"""

import argparse
import tempfile
from typing import Union, List

import PIL
import imageio
import numpy as np
import torch
from diffusers import CogVideoXPipeline


def export_to_video_imageio(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 8
) -> str:
    """
    Export the video frames to a video file using imageio lib to Avoid "green screen" issue (for example CogVideoX)
    """
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    return output_video_path


def generate_video(
    prompt: str,
    model_path: str,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").
    - dtype (torch.dtype): The data type for computation (default is torch.float16).
    """

    # Load the pre-trained CogVideoX pipeline with the specified precision (float16) and move it to the specified device
    # add device_map="balanced" in the from_pretrained function and remove
    # `pipe.enable_model_cpu_offload()` to enable Multi GPUs (2 or more and each one must have more than 20GB memory) inference.
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()

    # Encode the prompt to get the prompt embeddings
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,  # The textual description for video generation
        negative_prompt=None,  # The negative prompt to guide the video generation
        do_classifier_free_guidance=True,  # Whether to use classifier-free guidance
        num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
        max_sequence_length=226,  # Maximum length of the sequence, must be 226
        device=device,  # Device to use for computation
        dtype=dtype,  # Data type for computation
    )

    # Generate the video frames using the pipeline
    video = pipe(
        num_inference_steps=num_inference_steps,  # Number of inference steps
        guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance
        prompt_embeds=prompt_embeds,  # Encoded prompt embeddings
        negative_prompt_embeds=torch.zeros_like(prompt_embeds),  # Not Supported negative prompt
    ).frames[0]

    # Export the generated frames to a video file. fps must be 8
    export_to_video_imageio(video, output_path, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-2b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to use for computation (e.g., 'cuda' or 'cpu')"
    )

    parser.add_argument(
        "--dtype", type=str, default="float16", help="The data type for computation (e.g., 'float16' or 'float32')"
    )

    args = parser.parse_args()

    # Convert dtype argument to torch.dtype, NOT suggest BF16.
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # main function to generate video.
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        device=args.device,
        dtype=dtype,
    )
