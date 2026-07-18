# Math-Informed RL for UAV Relay Positioning
# Dockerfile for Vast.ai deployment

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    tqdm

# Copy project files
COPY scripts/ /workspace/scripts/
COPY results/ /workspace/results/
COPY README.md /workspace/
COPY MATH_INFORMED_RL_PLAN.md /workspace/

# Create output directories
RUN mkdir -p /workspace/results/mi_rl \
    /workspace/models/mi_rl

# Set environment
ENV PYTHONPATH=/workspace/scripts:$PYTHONPATH

# Default command: run MI-RL training
CMD ["python", "/workspace/scripts/mi_rl/train_mi_rl.py", "--episodes", "2000", "--eval-interval", "100"]
