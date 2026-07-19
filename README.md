# Math-Informed RL for UAV Relay Positioning

**Core Thesis: "A little bit of math + a little bit of RL = better than either alone"**

Combining classical optimization theory with modern reinforcement learning for UAV relay positioning in 6G low-altitude wireless networks (LAWNs).

## Demo

![MI-RL Demo](results/mi_rl/simulation_outputs/mi_rl_demo.gif)

*UAV relay positioning comparison: Random → Analytical → SCA → MI-RL*

## Version History

- **v1.0**: VLA-based approach using TinyLlama-1.1B (archived)
- **v2.0**: Math-Informed RL using SCA-Guided Actor-Critic (current)

## Overview

This repository implements **Math-Informed Reinforcement Learning (MI-RL)** for UAV relay positioning, combining:

1. **Classical Optimization (SCA)**: Successive Convex Approximation with convergence guarantees
2. **Physics-Informed Features**: Channel model knowledge encoded in state representation
3. **Residual RL**: Learning corrections to classical baseline
4. **Lyapunov Safety**: Stability guarantees during exploration

### Key Innovation

Unlike pure ML approaches that ignore domain knowledge, or pure optimization that assumes perfect models, MI-RL:
- **Warm starts from SCA** → Sample efficient (100x fewer samples than pure RL)
- **Physics-informed loss** → Respects Shannon capacity bounds
- **Lyapunov safety** → Guaranteed stability during adaptation
- **Residual learning** → Only learns what math can't capture

## Methods Compared

| Method | Throughput | Latency | Key Property |
|--------|------------|---------|--------------|
| Analytical Baseline | 118.4 Mbps | <1 ms | Simple heuristic |
| MLP (Supervised) | 170.2 Mbps | ~1 ms | Fast but rigid |
| Pure SCA-50 | ~175 Mbps | ~50 ms | Optimal but slow |
| Pure RL (SAC) | ~140 Mbps | ~5 ms | Adaptive but inefficient |
| **SGAC (MI-RL)** | **~180 Mbps** | **~5 ms** | **Best of both** |

## Project Structure

```
vla_6g_tvt/
├── scripts/
│   ├── eval_common.py             # Core evaluation infrastructure
│   ├── classical/                 # Classical optimization
│   │   ├── sca_solver.py          # Successive Convex Approximation
│   │   └── analytical_gradients.py # Closed-form channel gradients
│   ├── mi_rl/                     # Math-Informed RL (main contribution)
│   │   ├── physics_features.py    # Physics-informed state (47 dims)
│   │   ├── sgac_agent.py          # SCA-Guided Actor-Critic
│   │   ├── lyapunov_layer.py      # Safety projection layer
│   │   └── train_mi_rl.py         # Main training script
│   └── baselines/                 # Comparison methods
│       ├── train_eval_mlp.py      # MLP baseline
│       ├── train_eval_drl.py      # Pure RL baselines (PPO)
│       └── train_eval_sac_td3.py  # SAC/TD3 implementations
└── results/                       # Evaluation results & model checkpoints
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy scipy
```

### 2. Train Math-Informed RL

```bash
cd scripts/mi_rl
python train_mi_rl.py --episodes 1000
```

### 3. Run Complete Comparison

```bash
python train_mi_rl.py --episodes 2000 --eval-interval 100
```

### 4. Test Classical SCA Alone

```bash
cd scripts/classical
python sca_solver.py
```

## AirSim Demo (Realistic UAV Visualization)

For a realistic 3D demonstration of MI-RL UAV relay positioning:

### Docker Images

| Image | Description | Docker Hub |
|-------|-------------|------------|
| `abdulmannan617/airsim-recorder:latest` | Base AirSim environment | [Link](https://hub.docker.com/r/abdulmannan617/airsim-recorder) |
| `abdulmannan617/mi-rl-6g-airsim:latest` | MI-RL demo scripts | [Link](https://hub.docker.com/r/abdulmannan617/mi-rl-6g-airsim) |

### System Configuration

The `airsim-recorder` image is built for the following configuration:

| Component | Version |
|-----------|---------|
| **Base Image** | `nvidia/cuda:12.2.0-runtime-ubuntu22.04` |
| **OS** | Ubuntu 22.04 LTS |
| **CUDA** | 12.2.0 |
| **NVIDIA Driver** | ≥ 525 (CUDA 12.x compatible) |
| **AirSim Binary** | AirSimNH v1.8.1 (~15 GB) |
| **Python** | 3.10 with airsim v1.8.1 |
| **Display** | Xvfb (headless at 1920×1080) |

**Hardware Requirements:**
- GPU: NVIDIA RTX 3060 or better (compute capability ≥ 7.0)
- Disk: ≥ 80 GB (image is ~15 GB plus environment)
- RAM: 16 GB minimum, 32 GB recommended

### Quick Deploy to Vast.ai

```bash
# 1. Install CLI and set API key
pip install vastai
vastai set api-key YOUR_API_KEY

# 2. Search for compatible GPU instance (driver >= 525, 80GB disk)
vastai search offers 'reliability > 0.95 disk_space >= 80 gpu_name in [RTX_3060,RTX_3070,RTX_4060]' -o 'dph_total'

# 3. Deploy the airsim-recorder container
vastai create instance OFFER_ID \
    --image abdulmannan617/airsim-recorder:latest \
    --disk 100 --ssh

# 4. SSH in and start AirSim (Vast.ai overrides Docker entrypoint)
ssh -p PORT root@HOST
/opt/entrypoint.sh

# 5. Wait for "AirSim ready" message (~60-90 seconds)

# 6. Run your recording script
python3 /workspace/my_recording.py

# 7. Download output
scp -P PORT root@HOST:/workspace/output.mp4 ./
```

### Run Locally (Requires NVIDIA GPU)

```bash
docker run --gpus all -v $(pwd)/output:/workspace \
    abdulmannan617/airsim-recorder:latest
```

See [`airsim_demo/README.md`](airsim_demo/README.md) for full documentation.

## Mathematical Foundation

### Successive Convex Approximation (SCA)

The relay positioning problem is non-convex:
```
max   Σᵢ log₂(1 + SNRᵢ(p))
s.t.  p ∈ [0,100]² × [10,40]
```

SCA iteratively linearizes and solves convex subproblems with convergence guarantees.

### Physics-Informed Features (47 dimensions)

- Position features (normalized)
- Distance features (BS-UAV, UAV-users)
- SNR features (bottleneck, margins)
- Gradient features (optimal direction from SCA)
- Topology features (user distribution)

### Lyapunov Safety Constraint

```
V(s_{t+1}) - V(s_t) ≤ -γ·V(s_t)
```

Ensures exponential stability during RL exploration.

## Architecture: SCA-Guided Actor-Critic

```
State → Physics Features → SCA Gradient → Actor Network → Lyapunov → Action
                                ↓                            ↑
                          Residual Net ────────────────────────
```

The policy learns: `action = SCA_direction × 0.3 + Residual × 5.0`

## References

1. Razaviyayn et al., "SCA Convergence Analysis" (2013)
2. Chow et al., "Lyapunov-based Safe Policy Optimization" (2019)
3. [Physics-Informed RL Survey](https://arxiv.org/abs/2309.01909)
4. [Successive Convexification for Trajectory](https://arxiv.org/abs/2404.16826)

## Citation

```bibtex
@article{khan2026mirl,
  title={Math-Informed Reinforcement Learning for UAV Relay Positioning in Low-Altitude THz Networks},
  author={Khan, Abdul Mannan},
  journal={IEEE Internet of Things Journal},
  year={2026}
}
```

## License

MIT License
