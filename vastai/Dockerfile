# VLA-6G Training Docker Image
# For use on Vast.ai or any GPU cloud provider

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="VLA-6G Project"
LABEL description="TinyLlama fine-tuning for UAV relay positioning"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts
COPY generate_2m_data.py .
COPY train_vla_2m.py .
COPY run_full_pipeline.sh .

# Make scripts executable
RUN chmod +x run_full_pipeline.sh

# Set environment variables for optimal GPU usage
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TRANSFORMERS_CACHE=/workspace/cache
ENV HF_HOME=/workspace/cache

# Create directories
RUN mkdir -p /workspace/data /workspace/models /workspace/cache

# Default command
CMD ["/bin/bash"]
