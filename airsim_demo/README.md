# MI-RL 6G UAV Relay - AirSim Demo

Realistic demonstration of Math-Informed Reinforcement Learning (MI-RL) for 6G UAV relay positioning using AirSim simulation.

## Demo Video Features

- **Photorealistic urban environment** using AirSim Neighborhood (AirSimNH)
- **Real-time UAV positioning** showing MI-RL optimization
- **Performance comparison** across 4 methods:
  - Random baseline: 91 Mbps
  - Analytical: 94 Mbps
  - SCA-20: 195 Mbps
  - **MI-RL: 196 Mbps** (+107.9% over MLP baseline)

## Docker Image

**Docker Hub:** `abdulmannan617/mi-rl-6g-airsim:latest`

### Quick Deploy to Vast.ai

```bash
# 1. Install Vast.ai CLI
pip install vastai
vastai set api-key YOUR_API_KEY

# 2. Find a GPU instance
vastai search offers 'gpu_ram >= 8 dph < 0.5' --raw

# 3. Create instance with the demo image
vastai create instance OFFER_ID \
    --image abdulmannan617/mi-rl-6g-airsim:latest \
    --disk 50 \
    --ssh

# 4. Wait for completion and download video
scp -P SSH_PORT root@HOST:/home/airsim/output/mi_rl_demo.mp4 ./
```

### Run Locally (Requires NVIDIA GPU)

```bash
# Build the image
docker build -t mi-rl-6g-airsim -f Dockerfile.mirl_airsim .

# Run the demo
docker run --gpus all -v $(pwd)/output:/home/airsim/output \
    mi-rl-6g-airsim --duration 60

# Video will be saved to ./output/mi_rl_demo.mp4
```

## Files

| File | Description |
|------|-------------|
| `Dockerfile.mirl_airsim` | Docker image configuration |
| `mirl_demo_script.py` | AirSim recording script |
| `entrypoint_airsim.sh` | Container startup script |
| `config/airsim_settings.json` | AirSim vehicle & camera settings |
| `deploy_mirl_airsim_vastai.sh` | Automated Vast.ai deployment |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MI-RL Demo Container                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   AirSimNH   │───▶│  MI-RL Demo  │───▶│   Video      │  │
│  │ (UE4 Env)   │    │   Script     │    │   Output     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AirSim API (Python) - Drone Control & Image Capture │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Performance Comparison (from paper)

| Method | Throughput (Mbps) | Latency (ms) | Notes |
|--------|------------------|--------------|-------|
| Random | 91.0 | 0.1 | Baseline |
| MLP-2K | 93.8 | 0.1 | Neural regression |
| Analytical | 94.0 | 0.1 | Closed-form |
| SCA-20 | 195.0 | 850 | Iterative optimization |
| **MI-RL** | **196.0** | **45** | Math-informed RL |

MI-RL achieves **+107.9% throughput** over MLP baseline while maintaining reasonable latency.

## Related Files

- Paper: `../paper/vla_6g_ieee_tvt_v15.tex`
- Results: `../results/mi_rl/`
- Training scripts: `../scripts/mi_rl/`

## License

This demo is part of the VLA-6G research project.
