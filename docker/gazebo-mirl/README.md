# MI-RL Demo Docker Container

Pre-built Docker container for rendering MI-RL (Math-Informed Reinforcement Learning) UAV relay positioning demo videos using Blender and Gazebo.

## Quick Start

### Pull from Docker Hub
```bash
docker pull abdulmannan617/gazebo-mirl-demo:latest
```

### Render Demo Video (Blender - Recommended)
```bash
docker run --gpus all \
    -v /tmp/videos:/workspace/videos \
    abdulmannan617/gazebo-mirl-demo:latest \
    /workspace/scripts/render_blender_demo.sh my_demo
```

### Interactive Mode
```bash
docker run --gpus all -it \
    -v /tmp/videos:/workspace/videos \
    abdulmannan617/gazebo-mirl-demo:latest
```

## Contents

- **Blender 3.0**: Fast Workbench rendering for animated videos
- **Gazebo Harmonic**: Physics simulation with GUI
- **Xvfb**: Virtual framebuffer for headless rendering
- **ffmpeg**: Video encoding

## Scripts

| Script | Description |
|--------|-------------|
| `render_blender_demo.sh` | Render 10-second demo video using Blender (recommended) |
| `record_full_demo.sh` | Record Gazebo simulation with animation |
| `animate_mirl_demo.py` | UAV animation for Gazebo |
| `blender_mirl_demo_fast.py` | Blender scene and animation script |

## Output Video

The demo video shows:
- Urban environment with buildings
- Base station with antenna
- UAV relay drone
- 5 ground users (colored markers)
- Signal beams (BS-UAV and UAV-Users)
- UAV optimization animation: random exploration → convergence → optimal position

## Building Locally

```bash
cd /home/it-services/ros2_ws/src/vla_6g_tvt/docker/gazebo-mirl
./build_and_push.sh
```

## For Vast.ai

```bash
# On Vast.ai instance:
docker pull abdulmannan617/gazebo-mirl-demo:latest
docker run --gpus all -it \
    -v /root/videos:/workspace/videos \
    abdulmannan617/gazebo-mirl-demo:latest \
    /workspace/scripts/render_blender_demo.sh mirl_demo

# Download /root/videos/mirl_demo.mp4
```

## GPU Requirements

- NVIDIA GPU with CUDA 12.2+ support
- nvidia-container-toolkit on host
- Blender uses Workbench (CPU-accelerated), Cycles for higher quality (GPU)
