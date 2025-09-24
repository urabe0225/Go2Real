FROM python:3.11.9-slim-bullseye

RUN apt-get update && apt-get install -y \
    git cmake \
    iputils-ping \
    net-tools \
    iproute2 \
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

COPY networkTest.py /workspace/networkTest.py
