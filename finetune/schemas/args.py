import datetime
import argparse
from typing import Dict, Any, Literal, List, Tuple
from pydantic import BaseModel, field_validator

from pathlib import Path


class Args(BaseModel):
    ########## Model ##########
    model_path: Path
    model_name: str
    model_type: Literal["i2v", "t2v"]
    training_type: Literal["lora", "sft"] = "lora"

    ########## Output ##########
    output_dir: Path = Path("train_results/{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now()))
    report_to: Literal["tensorboard", "wandb", "all"] | None = None
    tracker_name: str = "finetrainer-cogvideo"

    ########## Data ###########
    data_root: Path
    caption_column: Path
    image_column: Path | None = None
    video_column: Path

    ########## Training #########
    resume_from_checkpoint: Path | None = None

    seed: int | None = None
    train_epochs: int
    train_steps: int | None = None
    checkpointing_steps: int = 500
    checkpointing_limit: int = 10

    batch_size: int
    gradient_accumulation_steps: int = 1

    train_resolution: Tuple[int, int, int]  # shape: (frames, height, width)

    #### deprecated args: video_resolution_buckets
    # if use bucket for training, should not be None
    # Note1: At least one frame rate in the bucket must be less than or equal to the frame rate of any video in the dataset
    # Note2:  For cogvideox, cogvideox1.5
    #   The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #   The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # video_resolution_buckets: List[Tuple[int, int, int]] | None = None

    mixed_precision: Literal["no", "fp16", "bf16"]

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    enable_slicing: bool = True
    enable_tiling: bool = True
    nccl_timeout: int = 1800

    ########## Lora ##########
    rank: int = 128
    lora_alpha: int = 64
    target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]

    ########## Validation ##########
    do_validation: bool = False
    validation_steps: int | None = None  # if set, should be a multiple of checkpointing_steps
    validation_dir: Path | None     # if set do_validation, should not be None
    validation_prompts: str | None  # if set do_validation, should not be None
    validation_images: str | None   # if set do_validation and model_type == i2v, should not be None
    validation_videos: str | None   # if set do_validation and model_type == v2v, should not be None
    gen_fps: int = 15

    #### deprecated args: gen_video_resolution
    # 1. If set do_validation, should not be None
    # 2. Suggest selecting the bucket from `video_resolution_buckets` that is closest to the resolution you have chosen for fine-tuning
    #        or the resolution recommended by the model
    # 3. Note:  For cogvideox, cogvideox1.5
    #        The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #        The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # gen_video_resolution: Tuple[int, int, int] | None  # shape: (frames, height, width)


    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--training_type", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--nccl_timeout", type=int, required=True)
        parser.add_argument("--mixed_precision", type=str, required=True)
        parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
        parser.add_argument("--data_root", type=str, required=True)
        parser.add_argument("--caption_column", type=str, required=True)
        parser.add_argument("--video_column", type=str, required=True)
        parser.add_argument("--image_column", type=str)
        parser.add_argument("--train_resolution", type=str, required=True)
        parser.add_argument("--batch_size", type=int, required=True)
        parser.add_argument("--num_workers", type=int, required=True)
        parser.add_argument("--pin_memory", type=str, required=True)
        parser.add_argument("--report_to", type=str, required=True)
        parser.add_argument("--train_epochs", type=int, required=True)
        parser.add_argument("--checkpointing_steps", type=int, required=True)
        parser.add_argument("--checkpointing_limit", type=int, required=True)

        parser.add_argument("--do_validation", type=bool)
        parser.add_argument("--validation_steps", type=int)
        parser.add_argument("--validation_dir", type=str)
        parser.add_argument("--validation_prompts", type=str)
        parser.add_argument("--validation_images", type=str)
        parser.add_argument("--validation_videos", type=str)
        parser.add_argument("--gen_fps", type=int)

        parser.add_argument("--resume_from_checkpoint", type=str)

        args = parser.parse_args()
        
        # Convert video_resolution_buckets string to list of tuples
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))

        return cls(**vars(args))

    # @field_validator("...", mode="after")
    # def foo(cls, foobar):
    #     ...

    # @field_validator("...", mode="before")
    # def bar(cls, barbar):
    #     ...
