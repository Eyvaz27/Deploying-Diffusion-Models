# Description: This file is used to build the docker image for the model server
FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python-is-python3 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    wget \
    google-perftools \
    && rm -rf /var/lib/apt/lists/*
RUN echo "LD_PRELOAD=/usr/lib/libtcmalloc.so.4" | tee -a /etc/environment
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN sed -i 's/python3/python3.11/g' /usr/bin/pip3
RUN sed -i 's/python3/python3.11/g' /usr/bin/pip

# Set up the workspace
WORKDIR /workspace/
# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_MODULE_LOADING LAZY
ENV LOG_VERBOSE 0

# TODO: Install the requirements file
RUN cd /workspace && mkdir diffusion && cd /workspace
COPY requirements.txt /workspace/diffusion/requirements.txt
RUN python -m pip install --no-cache-dir -r /workspace/diffusion/requirements.txt

# TODO: Copy the source code
COPY config.yaml /workspace/diffusion/config.yaml
COPY pipeline.py /workspace/diffusion/pipeline.py
COPY server.py /workspace/diffusion/server.py
COPY client.py /workspace/diffusion/client.py

# TODO: Run the server
# EXPOSE 8000

# WORKDIR /workspace/diffusion/
# ENTRYPOINT ["python", "-m", "server"]