# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from cog import BasePredictor, Input, Path


MODEL_CACHE = "model_cache"
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

        # model_id: THUDM/CogVideoX-5b
        self.pipe = CogVideoXPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance.",
        ),
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

        video = self.pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]

        out_path = "/tmp/out.mp4"

        export_to_video(video, out_path, fps=8)
        return Path(out_path)
