FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

SHELL ["/bin/bash", "-c"]


RUN apt-get update && apt-get upgrade -y
RUN apt install git -y \
&& apt install python3 -y \
&& apt install python3-pip -y \
&& apt install git -y \
&& apt install python-is-python3 -y \
&& apt install python3-tk -y \
&& apt install ffmpeg libsm6 libxext6  -y \ 
&& apt install git -y


WORKDIR /workspace

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
&& pip install opencv-python


RUN pip install SwissArmyTransformer>=0.2.9 icetk gifmaker
