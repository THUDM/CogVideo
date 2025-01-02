import os
import logging
import math
import json

import torch
import transformers
import diffusers
import wandb

from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from accelerate.logging import get_logger
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
    gather_object,
)

from diffusers.optimization import get_scheduler
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from finetune.schemas import Args, State, Components
from finetune.utils import (
    unwrap_model, cast_training_params,
    get_optimizer,

    get_memory_statistics,
    free_memory,

    get_latest_ckpt_path_to_resume_from,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_intermediate_ckpt_path,

    string_to_filename
)
from finetune.datasets import I2VDatasetWithResize, T2VDatasetWithResize
from finetune.datasets.utils import (
    load_prompts, load_images, load_videos,
    preprocess_image_with_resize, preprocess_video_with_resize
)

from finetune.constants import LOG_NAME, LOG_LEVEL


logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class Trainer:

    def __init__(self, args: Args) -> None:
        self.args  = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2]
        )

        self.components = Components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        # Initialize model components
        self.components = self.load_components()

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        if self.args.model_type == "i2v":
            self.dataset = I2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                encode_video_fn=self.encode_video,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width
            )
        elif self.args.model_type == "t2v":
            self.dataset = T2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                encode_video_fn=self.encode_video,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE for encoding
        self.components.vae = self.components.vae.to(self.accelerator.device)
        self.components.vae.requires_grad_(False)

        # Precompute latent for video
        logger.info("Precomputing latent for video ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        for _ in tmp_data_loader: ...
        logger.info("Precomputing latent for video ... Done")

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True
        )


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For now only lora is supported
        for attr_name, component in vars(self.components).items():
            if hasattr(component, 'requires_grad_'):
                component.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        self.__move_components_to_device()

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

        transformer_lora_config = LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.lora_alpha,
            init_lora_weights=True,
            target_modules=self.args.target_modules,
        )
        self.components.transformer.add_adapter(transformer_lora_config)
        self.__prepare_saving_loading_hooks(transformer_lora_config)

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([self.components.transformer], dtype=torch.float32)

        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.parameters()))
        transformer_parameters_with_lr = {
            "params": transformer_lora_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in transformer_lora_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch
    
    def prepare_for_validation(self):
        validation_prompts = load_prompts(self.args.validation_dir / self.args.validation_prompts)

        if self.args.validation_images is not None:
            validation_images = load_images(self.args.validation_dir / self.args.validation_images)
        else:
            validation_images = [None] * len(validation_prompts)

        if self.args.validation_videos is not None:
            validation_videos = load_videos(self.args.validation_dir / self.args.validation_videos)
        else:
            validation_videos = [None] * len(validation_prompts)

        self.state.validation_prompts = validation_prompts
        self.state.validation_images = validation_images
        self.state.validation_videos = validation_videos

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        self.accelerator.init_trackers(tracker_name, config=self.args.model_dump())

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                # Maybe run validation
                should_run_validation = (
                    self.args.do_validation
                    and global_step % self.args.validation_steps == 0
                )
                if should_run_validation:
                    self.validate(global_step)

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        all_processes_artifacts = []
        for i in range(num_validation_samples):
            # Skip current validation on all processes but one
            if i % accelerator.num_processes != accelerator.process_index:
                continue

            prompt = self.state.validation_prompts[i]
            image = self.state.validation_images[i]
            video = self.state.validation_videos[i]

            if image is not None:
                image = preprocess_image_with_resize(
                    image, self.state.train_height, self.state.train_width
                )
                # Convert image tensor (C, H, W) to PIL images
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)

            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = (video * 255).round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1,2,0).cpu().numpy()) for frame in video]

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            validation_artifacts = self.validation_step({
                "prompt": prompt,
                "image": image,
                "video": video
            })
            prompt_filename = string_to_filename(prompt)[:25]
            artifacts = {
                "image": {"type": "image", "value": image},
                "video": {"type": "video", "value": video},
            }
            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update({f"artifact_{i}": {"type": artifact_type, "value": artifact_value}})
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}.{extension}"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)

                if artifact_type == "image":
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = wandb.Image(filename)
                elif artifact_type == "video":
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                    artifact_value = wandb.Video(filename, caption=prompt)

                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
                    video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
                    tracker.log(
                        {
                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                        },
                        step=step,
                    )

        accelerator.wait_for_everyone()

        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        self.components.transformer.train()

    def fit(self):
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train()

    def collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError
    
    def load_components(self) -> Components:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        raise NotImplementedError
    
    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        raise NotImplementedError

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self):
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, 'to'):
                setattr(self.components, name, component.to(self.accelerator.device))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(
                            f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}"
                        )
            else:
                transformer_ = unwrap_model(self.accelerator, self.components.transformer).__class__.from_pretrained(
                    self.args.model_path, subfolder="transformer"
                )
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if self.args.mixed_precision == "fp16":
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params([transformer_])

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if must_save or global_step % self.args.checkpointing_steps == 0:
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path)
