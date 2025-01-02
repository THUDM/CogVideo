from ..cogvideox_t2v.lora_trainer import CogVideoXT2VLoraTrainer
from ..utils import register


class CogVideoX1dot5T2VLoraTrainer(CogVideoXT2VLoraTrainer):
    pass


register("cogvideox1.5-t2v", "lora", CogVideoX1dot5T2VLoraTrainer)
