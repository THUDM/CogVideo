from ..cogvideox_t2v.sft_trainer import CogVideoXT2VSftTrainer
from ..utils import register


class CogVideoX1dot5T2VSftTrainer(CogVideoXT2VSftTrainer):
    pass


register("cogvideox1.5-t2v", "sft", CogVideoX1dot5T2VSftTrainer)
