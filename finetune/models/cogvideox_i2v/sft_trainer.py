from ..cogvideox_i2v.lora_trainer import CogVideoXI2VLoraTrainer
from ..utils import register


class CogVideoXI2VSftTrainer(CogVideoXI2VLoraTrainer):
    pass


register("cogvideox-i2v", "sft", CogVideoXI2VSftTrainer)
