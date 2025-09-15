ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

RUN apt-get update && apt-get install -y \
    wget \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1-mesa-glx \
    libglfw3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pythonのインストール
RUN pip install mujoco gym numpy

WORKDIR /workspace

RUN wget https://github.com/google-deepmind/mujoco/releases/download/3.3.5/mujoco-3.3.5-linux-x86_64.tar.gz && \
    tar -xzvf mujoco-3.3.5-linux-x86_64.tar.gz && \
    mv mujoco-3.3.5 /opt/mujoco

COPY mujoco_test.py /workspace/mujoco_test.py

#CMD ["/usr/bin/xeyes"]
