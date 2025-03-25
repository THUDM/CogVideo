"""
The original experimental code for this project can be found at:

https://gist.github.com/a-r-r-o-w/d070cce059ab4ceab3a9f289ff83c69c

By using this code, description prompts will be generated through a local large language model, and images will be
generated using the black-forest-labs/FLUX.1-dev model, followed by video generation via CogVideoX.
The entire process utilizes open-source solutions, without the need for any API keys.

You can use the generate.sh file in the same folder to automate running this code
for batch generation of videos and images.

bash generate.sh

"""

import argparse
import gc
import json
import os
import pathlib
import random
from typing import Any, Dict

from transformers import AutoTokenizer

os.environ["TORCH_LOGS"] = "+dynamo,recompiles,graph_breaks"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

import numpy as np
import torch
import transformers
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler, DiffusionPipeline
from diffusers.utils.logging import get_logger
from diffusers.utils import export_to_video

torch.set_float32_matmul_precision("high")

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe.

For example, if you respond with "A beautiful morning in the woods with the sun peaking through the trees", the video generation model will create a video of exactly as described. You task is to summarize the descriptions of videos provided to by users, and create details prompts to feed into the generative model.

There are a few rules to follow:
- You will only ever output a single video description per request.
- If the user mentions to summarize the prompt in [X] words, make sure to not exceed the limit.

You responses should just be the video generation prompt. Here are examples:
- “A lone figure stands on a city rooftop at night, gazing up at the full moon. The moon glows brightly, casting a gentle light over the quiet cityscape. Below, the windows of countless homes shine with warm lights, creating a contrast between the bustling life below and the peaceful solitude above. The scene captures the essence of the Mid-Autumn Festival, where despite the distance, the figure feels connected to loved ones through the shared beauty of the moonlit sky.”
- "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
- "A street artist, clad in a worn-out denim jacket and a colorful banana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall"
""".strip()

USER_PROMPT = """
Could you generate a prompt for a video generation model?
Please limit the prompt to [{0}] words.
""".strip()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_videos",
        type=int,
        default=5,
        help="Number of unique videos you would like to generate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX-5B",
        help="The path of Image2Video CogVideoX-5B",
    )
    parser.add_argument(
        "--caption_generator_model_id",
        type=str,
        default="THUDM/glm-4-9b-chat",
        help="Caption generation model. default GLM-4-9B",
    )
    parser.add_argument(
        "--caption_generator_cache_dir",
        type=str,
        default=None,
        help="Cache directory for caption generation model.",
    )
    parser.add_argument(
        "--image_generator_model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Image generation model.",
    )
    parser.add_argument(
        "--image_generator_cache_dir",
        type=str,
        default=None,
        help="Cache directory for image generation model.",
    )
    parser.add_argument(
        "--image_generator_num_inference_steps",
        type=int,
        default=50,
        help="Caption generation model.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7, help="Guidance scale to be use for generation."
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        help="Whether or not to use cosine dynamic guidance for generation [Recommended].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Location where generated images and videos should be stored.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether or not to compile the transformer of image and video generators.",
    )
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Whether or not to use VAE tiling when encoding/decoding.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    return parser.parse_args()


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()


@torch.no_grad()
def main(args: Dict[str, Any]) -> None:
    output_dir = pathlib.Path(args.output_dir)
    os.makedirs(output_dir.as_posix(), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    reset_memory()
    tokenizer = AutoTokenizer.from_pretrained(
        args.caption_generator_model_id, trust_remote_code=True
    )
    caption_generator = transformers.pipeline(
        "text-generation",
        model=args.caption_generator_model_id,
        device_map="auto",
        model_kwargs={
            "local_files_only": True,
            "cache_dir": args.caption_generator_cache_dir,
            "torch_dtype": torch.bfloat16,
        },
        trust_remote_code=True,
        tokenizer=tokenizer,
    )

    captions = []
    for i in range(args.num_videos):
        num_words = random.choice([50, 75, 100])
        user_prompt = USER_PROMPT.format(num_words)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        outputs = caption_generator(messages, max_new_tokens=226)
        caption = outputs[0]["generated_text"][-1]["content"]
        if caption.startswith("\"") and caption.endswith("\""):
            caption = caption[1:-1]
        captions.append(caption)
        logger.info(f"Generated caption: {caption}")

    with open(output_dir / "captions.json", "w") as file:
        json.dump(captions, file)

    del caption_generator
    reset_memory()

    image_generator = DiffusionPipeline.from_pretrained(
        args.image_generator_model_id,
        cache_dir=args.image_generator_cache_dir,
        torch_dtype=torch.bfloat16,
    )
    image_generator.to("cuda")

    if args.compile:
        image_generator.transformer = torch.compile(
            image_generator.transformer, mode="max-autotune", fullgraph=True
        )

    if args.enable_vae_tiling:
        image_generator.vae.enable_tiling()

    images = []
    for index, caption in enumerate(captions):
        image = image_generator(
            prompt=caption,
            height=480,
            width=720,
            num_inference_steps=args.image_generator_num_inference_steps,
            guidance_scale=3.5,
        ).images[0]
        filename = (
            caption[:25].replace(".", "_").replace("'", "_").replace('"', "_").replace(",", "_")
        )
        image.save(output_dir / f"{index}_{filename}.png")
        images.append(image)

    del image_generator
    reset_memory()

    video_generator = CogVideoXImageToVideoPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to("cuda")
    video_generator.scheduler = CogVideoXDPMScheduler.from_config(
        video_generator.scheduler.config, timestep_spacing="trailing"
    )

    if args.compile:
        video_generator.transformer = torch.compile(
            video_generator.transformer, mode="max-autotune", fullgraph=True
        )

    if args.enable_vae_tiling:
        video_generator.vae.enable_tiling()

    generator = torch.Generator().manual_seed(args.seed)
    for index, (caption, image) in enumerate(zip(captions, images)):
        video = video_generator(
            image=image,
            prompt=caption,
            height=480,
            width=720,
            num_frames=49,
            num_inference_steps=50,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=args.use_dynamic_cfg,
            generator=generator,
        ).frames[0]
        filename = (
            caption[:25].replace(".", "_").replace("'", "_").replace('"', "_").replace(",", "_")
        )
        export_to_video(video, output_dir / f"{index}_{filename}.mp4", fps=8)


if __name__ == "__main__":
    args = get_args()
    main(args)
