"""
This script demonstrates how to generate a video from a text prompt using CogVideoX with quantization.

Note:

Must install the `torchao`，`torch` library FROM SOURCE to use the quantization feature.
Only NVIDIA GPUs like H100 or higher are supported om FP-8 quantization.

ALL quantization schemes must use with NVIDIA GPUs.

# Run the script:

python cli_demo_quantization.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-2b --quantization_scheme fp8 --dtype float16
python cli_demo_quantization.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --quantization_scheme fp8 --dtype bfloat16

"""

import argparse
import os
import torch
import torch._dynamo
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
)
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only
from torchao.float8.inference import ActivationCasting, QuantConfig, quantize_to_float8

os.environ["TORCH_LOGS"] = "+dynamo,output_code,graph_breaks,recompiles"
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


def quantize_model(part, quantization_scheme):
    if quantization_scheme == "int8":
        quantize_(part, int8_weight_only())
    elif quantization_scheme == "fp8":
        quantize_to_float8(part, QuantConfig(ActivationCasting.DYNAMIC))
    return part


def generate_video(
    prompt: str,
    model_path: str,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    quantization_scheme: str = "fp8",
    dtype: torch.dtype = torch.bfloat16,
    num_frames: int = 81,
    fps: int = 8,
    seed: int = 42,
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
    - quantization_scheme (str): The quantization scheme to use ('int8', 'fp8').
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    """
    text_encoder = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder = quantize_model(part=text_encoder, quantization_scheme=quantization_scheme)
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype
    )
    transformer = quantize_model(part=transformer, quantization_scheme=quantization_scheme)
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    vae = quantize_model(part=vae, quantization_scheme=quantization_scheme)
    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=dtype,
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).frames[0]

    export_to_video(video, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The description of the video to be generated"
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="Path of the pre-trained model"
    )
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="Path to save generated video"
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--num_videos_per_prompt", type=int, default=1, help="Videos to generate per prompt"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="Data type (e.g., 'float16', 'bfloat16')"
    )
    parser.add_argument(
        "--quantization_scheme",
        type=str,
        default="fp8",
        choices=["int8", "fp8"],
        help="Quantization scheme",
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the video")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for output video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        quantization_scheme=args.quantization_scheme,
        dtype=dtype,
        num_frames=args.num_frames,
        fps=args.fps,
        seed=args.seed,
    )
