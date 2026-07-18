# Math-Informed RL for UAV Relay Positioning

**Core Thesis: "A little bit of math + a little bit of RL = better than either alone"**

Comparing classical optimization and learning paradigms for UAV relay positioning in 6G low-altitude wireless networks (LAWNs).

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
│   ├── classical/                 # Classical optimization
│   │   ├── sca_solver.py          # Successive Convex Approximation
│   │   └── analytical_gradients.py # Closed-form channel gradients
│   ├── mi_rl/                     # Math-Informed RL
│   │   ├── physics_features.py    # Physics-informed state
│   │   ├── sgac_agent.py          # SCA-Guided Actor-Critic
│   │   ├── lyapunov_layer.py      # Safety projection layer
│   │   └── train_mi_rl.py         # Main training script
│   ├── baselines/                 # Comparison methods
│   │   ├── train_eval_mlp.py      # MLP baseline
│   │   ├── train_eval_drl.py      # Pure RL baselines
│   │   └── train_eval_sac.py      # SAC implementation
│   └── eval_common.py             # Shared utilities
├── models/                        # Trained checkpoints
├── results/                       # Evaluation results
└── paper/                         # IEEE publication
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
