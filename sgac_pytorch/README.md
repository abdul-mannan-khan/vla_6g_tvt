# Multi-UAV Agentic Coordination for 6G IoT Networks

**Emergent Multi-Agent Coordination for UAV Relay Positioning**

## Overview

This project implements a multi-agent reinforcement learning system where multiple UAV agents:
- **Negotiate** coverage areas through learned communication protocols
- **Coordinate** without centralized control (emergent behavior)
- **Adapt** to failures gracefully (fault tolerance)
- **Scale** to large fleets (decentralized architecture)

## Why Agentic AI over Classical Optimization?

| Feature | Centralized SCA | Multi-UAV Agents |
|---------|-----------------|------------------|
| Scalability | O(n³) complexity | O(n) linear |
| Fault Tolerance | Single point of failure | Emergent recovery |
| Adaptation | Full recomputation | Local adjustment |
| Communication | Global state required | Local messages only |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-UAV SYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│   UAV-1 Agent         UAV-2 Agent         UAV-3 Agent      │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐      │
│  │ Perceive │        │ Perceive │        │ Perceive │      │
│  │ Decide   │◀──────▶│ Decide   │◀──────▶│ Decide   │      │
│  │ Act      │  msgs  │ Act      │  msgs  │ Act      │      │
│  └──────────┘        └──────────┘        └──────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
sgac_pytorch/
├── core/
│   ├── multi_uav_agents.py   # Multi-agent system
│   └── environment_dynamic.py # Dynamic environment with mobility
├── learning/                  # MARL training (TODO)
├── evaluation/                # Metrics and experiments (TODO)
└── archive/                   # Legacy single-UAV code
```

## Key Features

1. **Emergent Coordination**: Agents learn to divide coverage without central planner
2. **Fault Tolerance**: When a UAV fails, others automatically adapt
3. **Dynamic Adaptation**: Handles moving users, unlike static SCA
4. **Scalable**: Adding UAVs doesn't require algorithm changes

## Quick Start

```python
from core.multi_uav_agents import MultiUAVEnvironment, UAVAgent

# Create environment
env = MultiUAVEnvironment(num_uavs=3, num_users=10)

# Run episode
obs = env.reset()
for step in range(100):
    actions = {i: agent.decide(obs[i]) for i, agent in enumerate(agents)}
    obs, rewards, done, info = env.step(actions)
```

## Experiments (Planned)

1. **Coverage vs # UAVs**: Scalability analysis
2. **Failure Recovery**: Time to restore coverage after UAV loss
3. **Overlap Minimization**: Coordination efficiency
4. **Comparison**: Agentic vs Centralized SCA

## Requirements

```
numpy
torch
```

## Citation

```bibtex
@article{multiuav_agentic_2026,
  title={Emergent Multi-Agent Coordination for UAV Relay Networks},
  author={...},
  journal={IEEE IoT Journal},
  year={2026}
}
```
