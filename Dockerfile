FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ARG FACEFUSION_VERSION=3.3.2
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory
WORKDIR /facefusion

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python-is-python3 \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default explicitly (some CUDA containers use python3.8+)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip if not available
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Clone your modified FaceFusion repo
RUN git clone https://github.com/dartv123/facefusion.git --branch master --single-branch .

# Install FaceFusion dependencies (no conda, just pip and CUDA ONNX)
RUN python install.py --onnxruntime cuda --skip-conda

# Optional: expose Gradio default port
EXPOSE 7860

# Default command (adjust to FaceFusion's actual launch script or CLI)
CMD ["python3", "facefusion.py", "run", "--execution-thread-count", "4", "--execution-providers", "cuda"]
