import torch

from typing_extensions import override
from typing import Any, Dict, List, Tuple
from PIL import Image

from transformers import AutoTokenizer, T5EncoderModel

from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers import (
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

from finetune.trainer import Trainer
from finetune.schemas import Components
from finetune.utils import unwrap_model
from ..utils import register


class CogVideoXI2VLoraTrainer(Trainer):

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )

        components.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path, subfolder="vae"
        )

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        return components

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {
            "encoded_videos": [],
            "prompt_token_ids": [],
            "images": []
        }
        
        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt = sample["prompt"]
            image = sample["image"]

            # tokenize prompt
            text_inputs = self.components.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.state.transformer_config.max_text_seq_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            ret["encoded_videos"].append(encoded_video)
            ret["prompt_token_ids"].append(text_input_ids[0])
            ret["images"].append(image)
    
        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_token_ids"] = torch.stack(ret["prompt_token_ids"])
        ret["images"] = torch.stack(ret["images"])

        return ret
    
    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_token_ids = batch["prompt_token_ids"]
        latent = batch["encoded_videos"]
        images = batch["images"]

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        prompt_embeds = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = self.components.vae.encode(noisy_images).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        ofs_emb = None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any]
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]

        pipe = self.components.pipeline_cls(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler
        )
        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            image=image,
            generator=self.state.generator
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-i2v", "lora", CogVideoXI2VLoraTrainer)