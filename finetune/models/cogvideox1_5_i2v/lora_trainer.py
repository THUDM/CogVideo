from ..cogvideox_i2v.lora_trainer import CogVideoXI2VLoraTrainer
from ..utils import register


class CogVideoX1_5I2VLoraTrainer(CogVideoXI2VLoraTrainer):
    pass


register("cogvideox1.5-i2v", "lora", CogVideoX1_5I2VLoraTrainer)
