import dotmap
from diffusers import CogVideoXPipeline
from typing_extensions import override

from finetune.utils import unwrap_model

from ..cogvideox_t2v.lora_trainer import CogVideoXT2VLoraTrainer
from ..utils import register


class CogVideoXT2VSftTrainer(CogVideoXT2VLoraTrainer):
    @override
    def initialize_pipeline(self) -> CogVideoXPipeline:
        origin_model = unwrap_model(self.accelerator, self.components.transformer)
        self.components.transformer.config.update(origin_model.config)
        self.components.transformer.config = dotmap.DotMap(self.components.transformer.config)
        pipe = CogVideoXPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=self.components.transformer,
            scheduler=self.components.scheduler,
        )
        return pipe


register("cogvideox-t2v", "sft", CogVideoXT2VSftTrainer)
