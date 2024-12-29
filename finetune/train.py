import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from finetune.schemas import Args
from finetune.models.utils import get_model_cls


def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
