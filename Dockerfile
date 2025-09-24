ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS go2-base

RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    cmake \
    python3.10 \
    python3-pip \
    python3.10-dev \
    # X11関連のライブラリを追加
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libxss1 \
    libgconf-2-4 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    # OpenGL関連
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglapi-mesa \
    libglu1-mesa \
    # EGL関連
    libegl1-mesa \
    libegl1-mesa-dev \
    # その他必要なライブラリ
    xvfb \
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

RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
#RUN pip install --no-cache-dir genesis-world==0.3.3
RUN pip install --no-cache-dir tensorboard rsl-rl-lib==2.2.4

COPY ./scripts/go2_env.py /workspace/go2_env.py
COPY ./scripts/friction_env.py /workspace/friction_env.py

###############################################################################
FROM go2-base AS go2-train
COPY ./training/go2_train.py /workspace/go2_train.py
COPY ./training/friction_train.py /workspace/friction_train.py

###############################################################################
FROM go2-base AS go2-real

COPY ./sim2real/sim2real_walk.py /workspace/sim2real_walk.py
COPY ./sim2real/unitree_legged_const.py /workspace/unitree_legged_const.py
COPY ./sim2real/my_genesis_go2_env.py /workspace/my_genesis_go2_env.py
