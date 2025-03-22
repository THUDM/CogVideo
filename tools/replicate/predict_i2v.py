# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from cog import BasePredictor, Input, Path


MODEL_CACHE = "model_cache_i2v"
MODEL_URL = f"https://weights.replicate.delivery/default/THUDM/CogVideo/{MODEL_CACHE}.tar"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # model_id: THUDM/CogVideoX-5b-I2V
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            MODEL_CACHE, torch_dtype=torch.bfloat16
        ).to("cuda")

        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="Starry sky slowly rotating."),
        image: Path = Input(description="Input image"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=6
        ),
        num_frames: int = Input(description="Number of frames for the output video", default=49),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        img = load_image(image=str(image))

        video = self.pipe(
            prompt=prompt,
            image=img,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]

        out_path = "/tmp/out.mp4"

        export_to_video(video, out_path, fps=8)
        return Path(out_path)
