import dotmap
from diffusers import CogVideoXImageToVideoPipeline
from typing_extensions import override

from finetune.utils import unwrap_model

from ..cogvideox_i2v.lora_trainer import CogVideoXI2VLoraTrainer
from ..utils import register


class CogVideoXI2VSftTrainer(CogVideoXI2VLoraTrainer):
    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        origin_model = unwrap_model(self.accelerator, self.components.transformer)
        self.components.transformer.config.update(origin_model.config)
        self.components.transformer.config = dotmap.DotMap(self.components.transformer.config)
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=self.components.transformer,
            scheduler=self.components.scheduler,
        )
        return pipe


register("cogvideox-i2v", "sft", CogVideoXI2VSftTrainer)
