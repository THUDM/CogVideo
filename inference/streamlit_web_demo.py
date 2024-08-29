"""
This script is used to create a Streamlit web application for generating videos using the CogVideoX model.

Run the script using Streamlit:
    $ export OPENAI_API_KEY=your OpenAI Key or ZhiupAI Key
    $ export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/  # using with ZhipuAI, Not using this when using OpenAI
    $ streamlit run web_demo.py
"""

import base64
import json
import os
import time
from datetime import datetime
from typing import List

import imageio
import numpy as np
import streamlit as st
import torch
from convert_demo import convert_prompt
from diffusers import CogVideoXPipeline


model_path: str = "THUDM/CogVideoX-2b"


# Load the model at the start
@st.cache_resource
def load_model(model_path: str, dtype: torch.dtype, device: str) -> CogVideoXPipeline:
    """
    Load the CogVideoX model.

    Args:
    - model_path (str): Path to the model.
    - dtype (torch.dtype): Data type for model.
    - device (str): Device to load the model on.

    Returns:
    - CogVideoXPipeline: Loaded model pipeline.
    """
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    return pipe


# Define a function to generate video based on the provided prompt and model path
def generate_video(
    pipe: CogVideoXPipeline,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> List[np.ndarray]:
    """
    Generate a video based on the provided prompt and model path.

    Args:
    - pipe (CogVideoXPipeline): The pipeline for generating videos.
    - prompt (str): Text prompt for video generation.
    - num_inference_steps (int): Number of inference steps.
    - guidance_scale (float): Guidance scale for generation.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - device (str): Device to run the generation on.
    - dtype (torch.dtype): Data type for the model.

    Returns:
    - List[np.ndarray]: Generated video frames.
    """
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=226,
        device=device,
        dtype=dtype,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    # Generate video
    video = pipe(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=torch.zeros_like(prompt_embeds),
    ).frames[0]
    return video


def save_video(video: List[np.ndarray], path: str, fps: int = 8) -> None:
    """
    Save the generated video to a file.

    Args:
    - video (List[np.ndarray]): Video frames.
    - path (str): Path to save the video.
    - fps (int): Frames per second for the video.
    """
    # Remove the first frame
    video = video[1:]

    writer = imageio.get_writer(path, fps=fps, codec="libx264")
    for frame in video:
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    writer.close()


def save_metadata(
    prompt: str,
    converted_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_videos_per_prompt: int,
    path: str,
) -> None:
    """
    Save metadata to a JSON file.

    Args:
    - prompt (str): Original prompt.
    - converted_prompt (str): Converted prompt.
    - num_inference_steps (int): Number of inference steps.
    - guidance_scale (float): Guidance scale.
    - num_videos_per_prompt (int): Number of videos per prompt.
    - path (str): Path to save the metadata.
    """
    metadata = {
        "prompt": prompt,
        "converted_prompt": converted_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_videos_per_prompt": num_videos_per_prompt,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)


def main() -> None:
    """
    Main function to run the Streamlit web application.
    """
    st.set_page_config(page_title="CogVideoX-Demo", page_icon="🎥", layout="wide")
    st.write("# CogVideoX 🎥")
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    global pipe
    pipe = load_model(model_path, dtype, device)

    with st.sidebar:
        st.info("It will take some time to generate a video (~90 seconds per videos in 50 steps).", icon="ℹ️")
        num_inference_steps: int = st.number_input("Inference Steps", min_value=1, max_value=100, value=50)
        guidance_scale: float = st.number_input("Guidance Scale", min_value=0.0, max_value=20.0, value=6.0)
        num_videos_per_prompt: int = st.number_input("Videos per Prompt", min_value=1, max_value=10, value=1)

        share_links_container = st.empty()

    prompt: str = st.chat_input("Prompt")

    if prompt:
        # Not Necessary, Suggestions
        with st.spinner("Refining prompts..."):
            converted_prompt = convert_prompt(prompt=prompt, retry_times=1)
            if converted_prompt is None:
                st.error("Failed to Refining the prompt, Using origin one.")

        st.info(f"**Origin prompt:**  \n{prompt}  \n  \n**Convert prompt:**  \n{converted_prompt}")
        torch.cuda.empty_cache()

        with st.spinner("Generating Video..."):
            start_time = time.time()
            video_paths = []

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./output/{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            metadata_path = os.path.join(output_dir, "config.json")
            save_metadata(
                prompt, converted_prompt, num_inference_steps, guidance_scale, num_videos_per_prompt, metadata_path
            )

            for i in range(num_videos_per_prompt):
                video_path = os.path.join(output_dir, f"output_{i + 1}.mp4")

                video = generate_video(
                    pipe, converted_prompt or prompt, num_inference_steps, guidance_scale, 1, device, dtype
                )
                save_video(video, video_path, fps=8)
                video_paths.append(video_path)
                with open(video_path, "rb") as video_file:
                    video_bytes: bytes = video_file.read()
                    st.video(video_bytes, autoplay=True, loop=True, format="video/mp4")
                torch.cuda.empty_cache()

            used_time: float = time.time() - start_time
            st.success(f"Videos generated in {used_time:.2f} seconds.")

            # Create download links in the sidebar
            with share_links_container:
                st.sidebar.write("### Download Links:")
                for video_path in video_paths:
                    video_name = os.path.basename(video_path)
                    with open(video_path, "rb") as f:
                        video_bytes: bytes = f.read()
                    b64_video = base64.b64encode(video_bytes).decode()
                    href = f'<a href="data:video/mp4;base64,{b64_video}" download="{video_name}">Download {video_name}</a>'
                    st.sidebar.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
