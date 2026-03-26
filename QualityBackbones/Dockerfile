FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.0" \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    build-essential \
    ninja-build \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging

RUN python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0

RUN python -m pip install psutil

RUN python -m pip install \
    "transformers>=4.56.0,<5" \
    "timm>=1.0.20" \
    "huggingface_hub>=0.24.0" \
    accelerate \
    peft \
    safetensors \
    pillow \
    numpy \
    pandas \
    pyarrow \
    tqdm \
    pyyaml \
    h5py \
    einops \
    sentencepiece \
    scipy \
    scikit-image \
    opencv-python \
    ftfy \
    regex \
    tensorboardX \
    torchsummary \
    numba \
    "lightning>=2.4,<3" \
    tensorboard \
    matplotlib

RUN python -c "import lightning, numba, timm, torch, transformers; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('numba', numba.__version__)"

WORKDIR /workspace
