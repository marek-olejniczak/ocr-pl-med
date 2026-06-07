# Base pinned loosely; confirm CUDA version against the server driver
# before the first build (nvidia-smi).
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

# opencv runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# code is bind-mounted at runtime (compose: .:/benchmark) - no rebuild on edits
WORKDIR /benchmark
