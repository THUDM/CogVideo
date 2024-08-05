# Contribution Guide

There may still be many incomplete aspects in this project.

We look forward to your contributions to the repository in the following areas. If you complete the work mentioned above
and are willing to submit a PR and share it with the community, upon review, we
will acknowledge your contribution on the project homepage.

## Model Algorithms

- Support for model quantization inference (Int4, Int8, etc. quantization engineering)
- Support for multi-card inference / model inference concurrency engineering
- Support for non-CUDA architecture inference devices

## Model Engineering / Secondary Development

- Model fine-tuning examples / best prompt practices
- Video super-resolution/frame interpolation for enhancing video generation quality.
- Any peripheral tools for the model
- Any minimal complete open-source projects using the CogVideoX open-source model

## Code Standards

Good code style is an art. We have prepared a `pyproject.toml` configuration file for the project to standardize code
style. You can organize the code according to the following specifications:

1. Install the `ruff` tool

```shell
pip install ruff
```

Then, run the `ruff` tool

```shell
ruff check tools sat inference
```

Check the code style. If there are issues, you can automatically fix them using the `ruff format` command.

```shell
ruff format tools sat inference
```

Once your code meets the standard, there should be no errors.

## Naming Conventions
1. Please use English names, do not use Pinyin or other language names. All comments should be in English.
2. Please strictly follow the PEP8 specification and use underscores to separate words. Do not use names like a, b, c.

