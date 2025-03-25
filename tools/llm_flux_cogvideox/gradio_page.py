import os
import gradio as gr
import gc
import random
import torch
import numpy as np
from PIL import Image
import transformers
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler, DiffusionPipeline
from diffusers.utils import export_to_video
from transformers import AutoTokenizer
from datetime import datetime, timedelta
import threading
import time
from moviepy import VideoFileClip

torch.set_float32_matmul_precision("high")

# Set default values
caption_generator_model_id = "/share/home/zyx/Models/Meta-Llama-3.1-8B-Instruct"
image_generator_model_id = "/share/home/zyx/Models/FLUX.1-dev"
video_generator_model_id = "/share/official_pretrains/hf_home/CogVideoX-5b-I2V"
seed = 1337

os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(caption_generator_model_id, trust_remote_code=True)
caption_generator = transformers.pipeline(
    "text-generation",
    model=caption_generator_model_id,
    device_map="balanced",
    model_kwargs={
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
    },
    trust_remote_code=True,
    tokenizer=tokenizer,
)

image_generator = DiffusionPipeline.from_pretrained(
    image_generator_model_id, torch_dtype=torch.bfloat16, device_map="balanced"
)
# image_generator.to("cuda")

video_generator = CogVideoXImageToVideoPipeline.from_pretrained(
    video_generator_model_id, torch_dtype=torch.bfloat16, device_map="balanced"
)

video_generator.vae.enable_slicing()
video_generator.vae.enable_tiling()

video_generator.scheduler = CogVideoXDPMScheduler.from_config(
    video_generator.scheduler.config, timestep_spacing="trailing"
)

# Define prompts
SYSTEM_PROMPT = """
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe.

For example, if you respond with "A beautiful morning in the woods with the sun peaking through the trees", the video generation model will create a video of exactly as described. Your task is to summarize the descriptions of videos provided by users and create detailed prompts to feed into the generative model.

There are a few rules to follow:
- You will only ever output a single video description per request.
- If the user mentions to summarize the prompt in [X] words, make sure not to exceed the limit.

Your responses should just be the video generation prompt. Here are examples:
- "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
- "A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart of the city, holding a can of spray paint, spray-painting a colorful bird on a mottled wall."
""".strip()

USER_PROMPT = """
Could you generate a prompt for a video generation model? Please limit the prompt to [{0}] words.
""".strip()


def generate_caption(prompt):
    num_words = random.choice([25, 50, 75, 100])
    user_prompt = USER_PROMPT.format(num_words)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt + "\n" + user_prompt},
    ]

    response = caption_generator(messages, max_new_tokens=226, return_full_text=False)
    caption = response[0]["generated_text"]
    if caption.startswith("\"") and caption.endswith("\""):
        caption = caption[1:-1]
    return caption


def generate_image(caption, progress=gr.Progress(track_tqdm=True)):
    image = image_generator(
        prompt=caption,
        height=480,
        width=720,
        num_inference_steps=30,
        guidance_scale=3.5,
    ).images[0]
    return image, image  # One for output One for State


def generate_video(caption, image, progress=gr.Progress(track_tqdm=True)):
    generator = torch.Generator().manual_seed(seed)
    video_frames = video_generator(
        image=image,
        prompt=caption,
        height=480,
        width=720,
        num_frames=49,
        num_inference_steps=50,
        guidance_scale=6,
        use_dynamic_cfg=True,
        generator=generator,
    ).frames[0]
    video_path = save_video(video_frames)
    gif_path = convert_to_gif(video_path)
    return video_path, gif_path


def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path, fps=8)
    return video_path


def convert_to_gif(video_path):
    clip = VideoFileClip(video_path)
    clip = clip.with_fps(8)
    clip = clip.resized(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               LLM + FLUX + CogVideoX-I2V Space 🤗
            </div>
    """)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=5)
            generate_caption_button = gr.Button("Generate Caption")
            caption = gr.Textbox(label="Caption", placeholder="Caption will appear here", lines=5)
            generate_image_button = gr.Button("Generate Image")
            image_output = gr.Image(label="Generated Image")
            state_image = gr.State()
            generate_caption_button.click(fn=generate_caption, inputs=prompt, outputs=caption)
            generate_image_button.click(
                fn=generate_image, inputs=caption, outputs=[image_output, state_image]
            )
        with gr.Column():
            video_output = gr.Video(label="Generated Video", width=720, height=480)
            download_video_button = gr.File(label="📥 Download Video", visible=False)
            download_gif_button = gr.File(label="📥 Download GIF", visible=False)
            generate_video_button = gr.Button("Generate Video from Image")
            generate_video_button.click(
                fn=generate_video,
                inputs=[caption, state_image],
                outputs=[video_output, download_gif_button],
            )

if __name__ == "__main__":
    demo.launch()
