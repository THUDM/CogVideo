"""
This script demonstrates how to generate a video from a text prompt using CogVideoX with 🤗Huggingface Diffusers Pipeline.

Note:
    This script requires the `diffusers>=0.31.0` library to be installed.

Run the script:
    $ python cli_demo.py --prompt "A girl ridding a bike." --model_path THUDM/CogVideoX-2b

"""

import gc
import argparse
import tempfile
from typing import Union, List

import PIL
import imageio
import numpy as np
import torch
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler

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
    - dtype (torch.dtype): The data type for computation (default is torch.float16).

    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (float16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for better results.
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model and reset the memory, enable tiling.
    pipe.enable_model_cpu_offload()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()

    # Using with diffusers branch `cogvideox-followup` to enable tiling. not support in `main` branch.
    # This will cost ONLY 12GB GPU memory.
    # pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps,so 48 frames and will plus 1 frame for the first frame.
    # for diffusers version `0.30.0`, this should be 48. and for `0.31.0` and after, this should be 49.
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
        num_inference_steps=num_inference_steps,  # Number of inference steps
        num_frames=49,
        guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance
        generator=torch.Generator().manual_seed(42),  # Set the seed for reproducibility
    ).frames[0]

    # 5. Export the generated frames to a video file. fps must be 8
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
        dtype=dtype,
    )
