# VLA-6G: UAV Relay Positioning for Low-Altitude Vehicular THz Networks

Comparing learning paradigms for UAV relay positioning in 6G low-altitude wireless networks (LAWNs).

## Overview

This repository contains code for training and evaluating different learning approaches for UAV relay positioning:

- **MLP**: Multilayer perceptron baseline (170.2 Mbps, sub-ms latency)
- **VLA**: Verbal Language Agent using TinyLlama-1.1B (134.0 Mbps, 822ms latency)
- **DRL**: Deep RL baselines (SAC, PPO, TD3)
- **Hybrid Controller**: Best practical performance (146.7 Mbps on obstacle scenarios)

## Key Results

| Method | Throughput | vs Analytical | Use Case |
|--------|------------|---------------|----------|
| Analytical | 118.4 Mbps | baseline | Simple heuristic |
| MLP (2K) | 170.2 Mbps | +43.8% | Fixed input format |
| VLA | 134.0 Mbps | +13.2% | Flexible text interface |
| Hybrid | 146.7 Mbps | +18.1% | Obstacle scenarios |

## Project Structure

```
vla_6g_tvt/
├── scripts/           # Training and evaluation scripts
│   ├── train_vla_optimized.py      # VLA/TinyLlama training
│   ├── train_eval_mlp.py           # MLP baseline
│   ├── train_eval_mlp_obstacle.py  # Obstacle-aware MLP
│   ├── train_eval_drl.py           # DRL baselines
│   └── eval_common.py              # Shared utilities
├── vastai/            # Vast.ai deployment for large-scale training
│   ├── generate_2m_data.py         # Generate 2M training samples
│   ├── train_vla_2m.py             # Train on 2M samples
│   └── Dockerfile                  # Docker environment
├── paper/             # IEEE paper (LaTeX)
├── models/            # Trained model weights
├── results/           # Evaluation results (JSON)
└── data/              # Training data
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers peft bitsandbytes scipy numpy
```

### 2. Train MLP Baseline

```bash
cd scripts
python train_eval_mlp.py
```

### 3. Train VLA (TinyLlama)

```bash
python train_vla_optimized.py
```

### 4. Evaluate on Obstacle Scenarios

```bash
python eval_final_comparison.py
```

## Large-Scale Training (2M samples)

For training on 2M samples, use Vast.ai:

```bash
cd vastai
# See README.md for Vast.ai deployment instructions
./run_full_pipeline.sh 2000000
```

## Citation

If you use this code, please cite:

```bibtex
@article{khan2026vla6g,
  title={Comparing Learning Paradigms for UAV Relay Positioning in Low-Altitude Vehicular THz Networks},
  author={Khan, Abdul Mannan},
  journal={IEEE Internet of Things Journal},
  year={2026}
}
```

## License

MIT License
