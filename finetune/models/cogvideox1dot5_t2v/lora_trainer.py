import torch

from typing_extensions import override
from typing import Any, Dict, List

from finetune.trainer import Trainer
from ..utils import register


class CogVideoX1dot5T2VLoraTrainer(Trainer):

    @override
    def collate_fn(self, samples: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError
    
    @override
    def load_components(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    @override
    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    @override
    def validate(self) -> None:
        raise NotImplementedError


register("cogvideox1.5-t2v", "lora", CogVideoX1dot5T2VLoraTrainer)
