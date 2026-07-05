# Match CUDA to the server driver before first build (like the other images).
# kraken needs Python <= 3.11 for its TF/torch stack on some versions - verify.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxml2 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# code is bind-mounted at runtime (compose: .:/benchmark) - no rebuild on edits
WORKDIR /benchmark
