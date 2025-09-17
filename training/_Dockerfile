ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    xvfb \
    x11-utils \
    libglib2.0-0 \
    libglib2.0-dev \
    libgtk-3-0 \
    libgtk-3-dev \
    libgdk-pixbuf2.0-0 \
    libgdk-pixbuf2.0-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libatk1.0-dev \
    libegl1-mesa \
    libegl1-mesa-dev \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds.git && \
    cd cyclonedds && \
    git checkout releases/0.10.x && \
    mkdir build install && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install

ENV CYCLONEDDS_HOME=/workspace/cyclonedds/install
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && \
    cd unitree_sdk2_python && \
    git submodule update --init --recursive && \
    pip install -e .

RUN pip install genesis-world==0.3.3
RUN pip install tensorboard rsl-rl-lib==2.2.4

RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git


# X11とOpenGLのサポートをインストール
RUN apt-get update && apt-get install -y \
    x11-apps \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxdamage1 \
    libxfixes3 \
    libxinerama1 \
    ca-certificates \
    fonts-liberation \
    libappindicator3-1 \
    libnss3 \
    lsb-release \
    xdg-utils

# 環境変数設定
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1