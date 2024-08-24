import gc
import math
import os
import random
import tempfile
import threading
import time

import gradio as gr
import numpy as np
import torch
from diffusers import CogVideoXPipeline
from datetime import datetime, timedelta

from diffusers.image_processor import VaeImageProcessor
from openai import OpenAI
import spaces
import moviepy.editor as mp
from typing import List, Union
import PIL
import utils
from rife_model import load_rife_model, rife_inference_with_latents

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.environ.get('MODEL_PATH', "THUDM/CogVideoX-2b")
UP_SCALE_MODEL_CKPT = os.environ.get('UP_SCALE_MODEL_CKPT', "")
RIFE_MODEL_PATH = os.environ.get('RIFE_MODEL_PATH', "")
pipe = CogVideoXPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(
    device)
pipe.enable_model_cpu_offload()

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_accumulated_memory_stats()
torch.cuda.reset_peak_memory_stats()

# pipe.vae.enable_tiling()
UP_SCALE_MODEL = None
RIFE_MODEL = None
sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""


def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = OpenAI()
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",
                 "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"'},
                {"role": "assistant",
                 "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance."},
                {"role": "user",
                 "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"'},
                {"role": "assistant",
                 "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field."},
                {"role": "user",
                 "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"'},
                {"role": "assistant",
                 "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background."},
                {"role": "user",
                 "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"'},
            ],
            model="glm-4-0520",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=200,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt


@spaces.GPU(duration=300)
def infer(
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int = -1,
        progress=gr.Progress(track_tqdm=True),
):
    torch.cuda.empty_cache()

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    video_pt = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pt",
        generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
    ).frames

    return (video_pt, seed)




def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace('.mp4', '.gif')
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        output_dir = './output'
        os.makedirs(output_dir, exist_ok=True)
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime < cutoff:
                    os.remove(file_path)
        time.sleep(600)  # Sleep for 10 minutes


threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX-5B Huggingface Space🤗
           </div>
           <div style="text-align: center;">
               <a href="https://huggingface.co/THUDM/CogVideoX-2B">🤗 2B Model Hub</a> |
               <a href="https://huggingface.co/THUDM/CogVideoX-5B">🤗 5B Model Hub</a> |
               <a href="https://github.com/THUDM/CogVideo">🌐 Github</a> |
               <a href="https://arxiv.org/pdf/2408.06072">📜 arxiv </a>
           </div>

           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experiential use only. 
            Users should strictly adhere to local laws and ethics.
            </div>
           """)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)

            with gr.Row():
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one.")
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")

            with gr.Column():
                gr.Markdown("**Optional Parameters** (default values are recommended)<br>"
                            "Increasing the number of inference steps will produce more detailed videos, but it will slow down the process.<br>"
                            "50 steps are recommended for most cases.<br>"
                            "For the 5B model, 50 steps will take approximately 350 seconds.")
                with gr.Row():
                    num_inference_steps = gr.Number(label="Inference Steps", value=50)
                    guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                with gr.Row():
                    seed_param = gr.Number(label="Inference Seed", value=-1)
                with gr.Row():
                    enable_scale = gr.Checkbox(label="Enable Upscale", value=False)
                    enable_rife = gr.Checkbox(label="Enable RIFE", value=False)
                generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                seed_text = gr.Number(label="seed", value=-1)

    gr.Markdown("""
    <table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
         <div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;">
               Demo Videos with 50 Inference Steps and 6.0 Guidance Scale.
         </div>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 1.2em;">
                <p>A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="100%" controls autoplay></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 1.2em;">
                <p>The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from its tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/9de41efd-d4d1-4095-aeda-246dd834e91d" width="100%" controls autoplay></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 1.2em;">
                <p>A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/941d6661-6a8d-4a1b-b912-59606f0b2841" width="100%" controls autoplay></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 1.2em;">
                <p>In the haunting backdrop of a war-torn city, where ruins and crumbled walls tell a story of devastation, a poignant close-up frames a young girl. Her face is smudged with ash, a silent testament to the chaos around her. Her eyes glistening with a mix of sorrow and resilience, capturing the raw emotion of a world that has lost its innocence to the ravages of conflict.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/938529c4-91ae-4f60-b96b-3c3947fa63cb" width="100%" controls autoplay></video>
            </td>
        </tr>
    </table>
    """)


    def generate(prompt, num_inference_steps, guidance_scale,seed_value, scale_status, rife_status, progress=gr.Progress(track_tqdm=True)):
        global UP_SCALE_MODEL, RIFE_MODEL
        if not UP_SCALE_MODEL and len(str(UP_SCALE_MODEL_CKPT).strip()) > 0:
            # Load the upscale model with progress tracking
            UP_SCALE_MODEL = utils.load_sd_upscale(UP_SCALE_MODEL_CKPT, device)
        if not RIFE_MODEL and len(str(RIFE_MODEL_PATH).strip()) > 0:
            # Load the RIFE model with progress tracking
            RIFE_MODEL = load_rife_model(RIFE_MODEL_PATH)

        latents, seed = infer(prompt, num_inference_steps, guidance_scale, seed=seed_value, progress=progress)
        if UP_SCALE_MODEL and scale_status:
            latents = utils.upscale_batch_and_concatenate(UP_SCALE_MODEL, latents, device)

        if RIFE_MODEL and rife_status:
            latents = rife_inference_with_latents(RIFE_MODEL, latents)

        batch_size = latents.shape[0]
        batch_video_frames = []
        for batch_idx in range(batch_size):
            pt_image = latents[batch_idx]
            pt_image = torch.stack(
                [pt_image[i] for i in range(pt_image.shape[0])]
            )

            image_np = VaeImageProcessor.pt_to_numpy(pt_image)  # (to [49, 512, 480, 3])

            image_pil = VaeImageProcessor.numpy_to_pil(image_np)

            batch_video_frames.append(image_pil)

        # fps (len(batch_video_frames[0])-1) /6
        video_path = utils.save_video(batch_video_frames[0], fps=math.ceil((len(batch_video_frames[0])-1) / 6))
        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)

        seed_update = gr.update(value=seed)
        return video_path, video_update, gif_update, seed_update


    def enhance_prompt_func(prompt):
        return convert_prompt(prompt, retry_times=1)


    generate_button.click(
        generate,
        inputs=[prompt, num_inference_steps, guidance_scale, seed_param, enable_scale, enable_rife],
        outputs=[video_output, download_video_button, download_gif_button, seed_text]
    )

    enhance_button.click(
        enhance_prompt_func,
        inputs=[prompt],
        outputs=[prompt]
    )

if __name__ == "__main__":
    demo.launch(server_port=7870)