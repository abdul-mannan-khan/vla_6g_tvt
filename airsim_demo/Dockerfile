# MI-RL 6G UAV Relay - AirSim Demo Container
# Custom environment for demonstrating Math-Informed RL UAV positioning

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    x11vnc \
    xvfb \
    fluxbox \
    novnc \
    websockify \
    supervisor \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    airsim \
    msgpack-rpc-python \
    torch \
    tqdm \
    Pillow

# Create workspace
WORKDIR /workspace

# Copy MI-RL code
COPY scripts/ /workspace/scripts/
COPY results/ /workspace/results/

# Copy AirSim demo files
COPY airsim_demo/ /workspace/airsim_demo/

# AirSim settings
RUN mkdir -p /root/Documents/AirSim
COPY airsim_demo/settings.json /root/Documents/AirSim/settings.json

# Setup display
ENV DISPLAY=:99
ENV RESOLUTION=1920x1080x24

# Supervisor config for services
COPY airsim_demo/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports
# 8080: noVNC web interface
# 5900: VNC
# 41451: AirSim API
EXPOSE 8080 5900 41451

# Entry point
COPY airsim_demo/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["demo"]
