# Plain python base on purpose: surya-ocr pins its own torch, and starting from
# the pytorch image left torch/torchvision mismatched ("torchvision::nms does
# not exist"). Letting pip resolve torch+torchvision together with surya gives a
# consistent CUDA set (PyPI linux wheels are CUDA-enabled; driver 580 supports them).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# code is bind-mounted at runtime (compose: .:/benchmark) - no rebuild on edits
WORKDIR /benchmark
