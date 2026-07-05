# Match CUDA to the server driver (nvidia-smi) before first build, like the
# ultralytics image. detectron2 is built from source against this torch/CUDA.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

# build tools + opencv runtime deps (detectron2 compiles CUDA/C++ ops)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# detectron2 from source; TORCH_CUDA_ARCH_LIST avoids needing a GPU at build
# time (RTX 3090 = 8.6). Verify the tag/arch on the server.
ENV TORCH_CUDA_ARCH_LIST="8.6"
RUN pip install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git'

# code is bind-mounted at runtime (compose: .:/benchmark) - no rebuild on edits
WORKDIR /benchmark
