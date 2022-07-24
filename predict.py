import os
from random import randint
import subprocess
import tempfile
import glob
import typing
from deep_translator import GoogleTranslator
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        subprocess.call("python setup.py install", cwd="/src/Image-Local-Attention", shell=True)
        self.translator = GoogleTranslator(source="en", target="zh-CN")

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        seed: int = Input(description="Seed (leave empty to use a random seed)", default=None, le=(2**32 - 1), ge=0),
        translate: bool = Input(
            description="Translate prompt from English to Simplified Chinese (required if not entering Chinese text)",
            default=True,
        ),
        # both_stages: bool = Input(
        #    description="Run both stages (uncheck to run only stage 1 for quicker results)", default=True
        # ),
        use_guidance: bool = Input(description="Use stage 1 guidance (recommended)", default=True),
    ) -> typing.List[Path]:
        if translate:
            prompt = self.translator.translate(prompt)
        workdir = tempfile.mkdtemp()
        os.makedirs(f"{workdir}/output")
        with open(f"{workdir}/input.txt", "w") as f:
            f.write(prompt)
        if seed is None:
            seed = randint(0, 2**32)
        args = [
            "python",
            "cogvideo_pipeline.py",
            "--input-source",
            f"{workdir}/input.txt",
            "--output-path",
            f"{workdir}/output",
            "--batch-size",
            "1",
            "--parallel-size",
            "1",
            "--guidance-alpha",
            "3.0",
            "--generate-frame-num",
            "4",
            "--tokenizer-type",
            "fake",
            "--mode",
            "inference",
            "--distributed-backend",
            "nccl",
            "--fp16",
            "--model-parallel-size",
            "1",
            "--temperature",
            "1.05",
            "--coglm-temperature",
            "0.89",
            "--top_k",
            "12",
            "--sandwich-ln",
            "--seed",
            str(seed),
            "--num-workers",
            "0",
            "--batch-size",
            "1",
            "--max-inference-batch-size",
            "8",
            "--both-stages",
        ]
        if use_guidance:
            args.append("--use-guidance-stage1")
        print(args)
        os.environ["SAT_HOME"] = "/sharefs/cogview-new"
        if subprocess.check_output(args, shell=False, cwd="/src"):
            output = glob.glob(f"{workdir}/output/**/*.gif")
            for f in output:
                yield Path(f)
