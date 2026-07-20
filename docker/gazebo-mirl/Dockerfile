# MI-RL Demo Docker Image with Interactive Remote Desktop
# Includes Gazebo Harmonic, Blender, VNC, and noVNC for browser access
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

LABEL maintainer="VLA-6G TVT Research"
LABEL description="MI-RL UAV relay positioning demo with remote desktop access"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Add Gazebo Harmonic repository
RUN apt-get update && apt-get install -y \
    lsb-release \
    wget \
    gnupg \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo Harmonic
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get update \
    && apt-get install -y gz-harmonic \
    && rm -rf /var/lib/apt/lists/*

# Install desktop, VNC, noVNC, and rendering tools
RUN apt-get update && apt-get install -y \
    blender \
    xvfb \
    x11vnc \
    tigervnc-standalone-server \
    tigervnc-common \
    novnc \
    websockify \
    openbox \
    xterm \
    supervisor \
    ffmpeg \
    mesa-utils \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libegl1-mesa \
    libgles2-mesa \
    vulkan-tools \
    libvulkan1 \
    python3 \
    python3-pip \
    git \
    dbus-x11 \
    libxkbcommon-x11-0 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Copy world files and scripts
COPY worlds/ /workspace/worlds/
COPY scripts/ /workspace/scripts/

# Environment variables
ENV DISPLAY=:1
ENV LIBGL_ALWAYS_SOFTWARE=0
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV MESA_LOADER_DRIVER_OVERRIDE=nvidia
ENV GZ_SIM_RESOURCE_PATH=/workspace/worlds
ENV VNC_PASSWORD=mirl2024
ENV NOVNC_PORT=6080
ENV VNC_PORT=5901

# Create directories
RUN mkdir -p /workspace/videos /root/.vnc /var/log/supervisor

# VNC password setup (default: mirl2024)
RUN echo "mirl2024" | vncpasswd -f > /root/.vnc/passwd && chmod 600 /root/.vnc/passwd

# VNC startup script
RUN echo '#!/bin/bash\nopenbox &\nexec xterm -geometry 100x30 -fa "DejaVu Sans Mono" -fs 11' > /root/.vnc/xstartup && \
    chmod +x /root/.vnc/xstartup

# Supervisor configuration for managing services
RUN echo '[supervisord]\nnodaemon=true\nlogfile=/var/log/supervisor/supervisord.log\n\n\
[program:vnc]\ncommand=vncserver :1 -geometry 1920x1080 -depth 24 -fg\nautorestart=true\n\n\
[program:novnc]\ncommand=websockify --web=/usr/share/novnc/ 6080 localhost:5901\nautorestart=true\n' > /etc/supervisor/conf.d/vnc.conf

# Entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports: VNC (5901), noVNC web (6080)
EXPOSE 5901 6080

ENTRYPOINT ["/entrypoint.sh"]
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
