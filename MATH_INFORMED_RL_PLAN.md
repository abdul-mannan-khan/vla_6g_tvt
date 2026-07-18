# Math-Informed Reinforcement Learning for UAV Relay Positioning

## Core Thesis

**"A little bit of math + a little bit of RL = better than either alone"**

Pure MLP regression achieves 170.2 Mbps but lacks adaptability. Pure RL struggles with sample efficiency. Classical optimization (SCA, BCD) has convergence guarantees but assumes perfect knowledge. We propose **Math-Informed RL (MI-RL)** that combines:

1. **Classical optimization structure** → Guides exploration, provides warm starts
2. **Physics-informed losses** → Encodes channel model into neural network training
3. **Lyapunov stability** → Guarantees safe actions during adaptation

## Architecture: SCA-Guided Actor-Critic (SGAC)

```
┌─────────────────────────────────────────────────────────────────┐
│                     SCA-Guided Actor-Critic                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐     ┌──────────────────┐     ┌────────────┐ │
│  │ State s_t     │────▶│ Physics-Informed │────▶│ SCA Warm   │ │
│  │ (positions,   │     │ Feature Encoder  │     │ Start π_0  │ │
│  │  channels)    │     └──────────────────┘     └─────┬──────┘ │
│  └───────────────┘                                    │        │
│                                                       ▼        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Actor Network (Policy π_θ)                    │ │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌───────────┐  │ │
│  │  │ SCA     │ + │ Residual│ = │ Final   │ → │ Lyapunov  │  │ │
│  │  │ Solution│   │ ΔAction │   │ Action  │   │ Projection│  │ │
│  │  └─────────┘   └─────────┘   └─────────┘   └───────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Critic Network (Value V_φ)                    │ │
│  │  Physics-informed value function with channel metrics      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Loss = L_RL + λ₁·L_physics + λ₂·L_Lyapunov                    │
└─────────────────────────────────────────────────────────────────┘
```

## Mathematical Foundation

### 1. Successive Convex Approximation (SCA)

The relay positioning problem is non-convex:
```
max   Σᵢ log₂(1 + SNRᵢ(p))
s.t.  p ∈ [0,100]² × [10,40]
```

SCA iteratively solves convex subproblems:
```
p^(k+1) = argmax   Σᵢ log₂(1 + SNRᵢ(p^(k))) + ∇SNRᵢ(p^(k))ᵀ(p - p^(k))
          s.t.     ||p - p^(k)|| ≤ Δ  (trust region)
```

**Convergence guarantee**: Every limit point is a stationary point (Razaviyayn et al., 2014).

### 2. Physics-Informed Loss

Encode THz channel physics directly in the loss:
```python
L_physics = ||predicted_throughput - shannon_capacity(predicted_SNR)||²
          + ||∇_p throughput - analytical_gradient||²
```

This ensures the network respects Shannon capacity bounds.

### 3. Lyapunov Stability Constraint

Define Lyapunov function V(s) = ||p - p_target||² + α·variance(throughputs)

Safety constraint:
```
V(s_{t+1}) - V(s_t) ≤ -γ·V(s_t)  (exponential stability)
```

Project actions to satisfy this constraint before execution.

## Implementation Plan

### Phase 1: Classical Optimization Baseline
- Implement SCA solver with convergence tracking
- Implement Block Coordinate Descent (BCD) for comparison
- Establish oracle performance upper bound

### Phase 2: Physics-Informed Feature Engineering
- Extract physics-meaningful features (path loss, absorption coefficients)
- Pre-compute gradients for warm starting
- Create channel-aware state representation

### Phase 3: SCA-Guided Actor-Critic
- Initialize policy from SCA solution
- Train residual policy for adaptation
- Add physics-informed regularization

### Phase 4: Lyapunov Safety Layer
- Implement Lyapunov projection layer
- Guarantee safe actions during exploration
- Validate stability under disturbances

### Phase 5: Evaluation & Comparison
- Compare: Pure SCA, Pure RL, MLP, MI-RL (ours)
- Test on: Static, Dynamic, Obstacle, Unknown scenarios
- Demonstrate: "Math + RL > either alone"

## Expected Results

| Method | Static | Dynamic | Obstacle | Unknown | Key Property |
|--------|--------|---------|----------|---------|--------------|
| Analytical | 118.4 | ~90 | ~80 | ~70 | Simple, no learning |
| MLP | 170.2 | ~130 | ~140 | ~100 | Fast but rigid |
| Pure SCA | ~175 | ~120 | ~120 | ~90 | Optimal but slow |
| Pure SAC | ~140 | ~145 | ~130 | ~135 | Adaptive but inefficient |
| **MI-RL (Ours)** | **~180** | **~165** | **~170** | **~160** | Best of both |

## Key Innovation

Unlike pure ML approaches that ignore domain knowledge, or pure optimization that assumes perfect models, MI-RL:

1. **Warm starts from SCA** → Sample efficient (100x fewer samples than pure RL)
2. **Physics-informed loss** → Respects Shannon capacity bounds
3. **Lyapunov safety** → Guaranteed stability during adaptation
4. **Residual learning** → Only learns what math can't capture

## References

1. [Successive Convexification for Trajectory Optimization](https://arxiv.org/abs/2404.16826) - Convergence guarantees
2. [Physics-Informed RL Survey](https://arxiv.org/abs/2309.01909) - PIRL methods
3. [Science-Informed Deep Learning for Wireless](https://arxiv.org/pdf/2407.07742) - Tutorial on integration
4. [Lyapunov-based Safe RL](https://arxiv.org/abs/2201.00451) - Safety constraints
5. [SCA for UAV Trajectory](https://www.mdpi.com/1999-5903/17/5/225) - UAV application

## File Structure After Overhaul

```
vla_6g_tvt/
├── scripts/
│   ├── classical/
│   │   ├── sca_solver.py           # Successive Convex Approximation
│   │   ├── bcd_solver.py           # Block Coordinate Descent
│   │   └── analytical_gradients.py # Closed-form channel gradients
│   ├── mi_rl/
│   │   ├── physics_features.py     # Physics-informed features
│   │   ├── sgac_agent.py           # SCA-Guided Actor-Critic
│   │   ├── lyapunov_layer.py       # Safety projection layer
│   │   └── train_mi_rl.py          # Main training script
│   ├── baselines/                  # Keep existing baselines
│   │   ├── train_eval_mlp.py
│   │   ├── train_eval_sac.py
│   │   └── train_eval_drl.py
│   └── eval_common.py              # Shared evaluation (unchanged)
├── models/
│   └── mi_rl/                      # New model checkpoints
└── results/
    └── mi_rl/                      # New results
```
