from ..cogvideox_t2v.lora_trainer import CogVideoXT2VLoraTrainer
from ..utils import register


class CogVideoXT2VSftTrainer(CogVideoXT2VLoraTrainer):
    pass


register("cogvideox-t2v", "sft", CogVideoXT2VSftTrainer)
