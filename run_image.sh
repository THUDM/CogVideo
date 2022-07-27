#!/bin/sh
docker run --rm --name cog -it --gpus all  -v "${PWD}":/workspace cog