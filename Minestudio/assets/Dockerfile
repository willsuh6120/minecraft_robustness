FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    gnutls-bin \
    openssh-client \
    libghc-x11-dev \
    gcc-multilib \
    g++-multilib \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    xvfb \
    mesa-utils \
    libegl1-mesa \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    unzip \
    openjdk-8-jdk

RUN pip install --upgrade pip &&\
    pip install MineStudio && \
    python -m minestudio.simulator.entry -y

CMD ["python", "-m", "minestudio.simulator.entry"]
