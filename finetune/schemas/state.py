import torch

from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, field_validator

class State(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    train_frames: int
    train_height: int
    train_width: int

    transformer_config: Dict[str, Any] = None

    weight_dtype: torch.dtype = torch.float32
    num_trainable_parameters: int = 0
    overwrote_max_train_steps: bool = False
    num_update_steps_per_epoch: int = 0
    total_batch_size_count: int = 0

    generator: torch.Generator | None = None

    validation_prompts: List[str] = []
    validation_images: List[Path | None] = []
    validation_videos: List[Path | None] = []
