# AirSim Demo Recovery State - July 19, 2026

## Current Status
Working on getting AirSim to run for MI-RL 6G demo video.

## Key Finding: Original Configuration
The `abdulmannan617/airsim-recorder` Docker image was built with:

| Component | Version |
|-----------|---------|
| Base Image | `nvidia/cuda:12.2.0-runtime-ubuntu22.04` |
| Ubuntu | 22.04 |
| CUDA | 12.2.0 |
| Required NVIDIA Driver | ≥ 525 |
| AirSim | AirSimNH v1.8.1 |

## Root Cause of Previous Crashes
The crash was caused by running AirSim incorrectly:
```
libc++abi: terminating with uncaught exception of type std::__1::ios_base::failure:
mkdir failed for path /root/Documents with errorno 13 and message Permission denied
```

**Solution:** Vast.ai overrides Docker entrypoint. Must manually run:
```bash
/opt/entrypoint.sh
```

This entrypoint:
1. Starts Xvfb on :99
2. Runs AirSim as `airsim` user (not root) using `runuser -u airsim`
3. Waits for API port 41451
4. Warms up renderer

## Files to Use
- **Docker Image:** `abdulmannan617/airsim-recorder:latest`
- **Entrypoint:** `/opt/entrypoint.sh` (MUST run manually on Vast.ai)
- **Settings:** `/home/airsim/Documents/AirSim/settings.json`
- **AirSim Binary:** `/opt/AirSim/AirSimNH/LinuxNoEditor/AirSimNH.sh`

## Correct Vast.ai Deployment Steps
1. Create instance with `abdulmannan617/airsim-recorder:latest`
2. SSH in: `ssh -p <port> root@<host>`
3. Run entrypoint: `/opt/entrypoint.sh`
4. Wait for "AirSim ready" (~60-90 seconds)
5. Upload and run demo script

## MI-RL Demo Script Location
`/home/it-services/ros2_ws/src/vla_6g_tvt/airsim_demo/mirl_demo_script.py`

## Trained Model Location
`/home/it-services/ros2_ws/src/vla_6g_tvt/results/mi_rl/checkpoints/checkpoint_latest.pt`

## Next Steps
1. Create new Vast.ai instance with RTX 3060+ ($0.06-0.10/hr)
2. Use `abdulmannan617/airsim-recorder:latest` image
3. Run `/opt/entrypoint.sh` manually (Vast.ai overrides entrypoint)
4. Upload and run mirl_demo_script.py
5. Download the recorded video

## Instance Requirements
- GPU: NVIDIA RTX 3060 or better
- Disk: ≥ 80 GB
- Driver: ≥ 525 (CUDA 12.x compatible)
- Network: ≥ 1 Gbps download

## Important Notes
- DO NOT destroy instances not created in this session
- Previous crashes were due to running AirSim as root instead of airsim user
- The image is ~15 GB, so allow time for it to load
