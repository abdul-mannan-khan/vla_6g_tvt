#!/usr/bin/env python3
"""
W3 Pre-emption: Train MLP on 2,000 samples (same data budget as VLA)
for a fair head-to-head comparison.
"""

import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (
    generate_scenarios, compute_channel_metrics,
    extract_features_from_raw, extract_features,
    get_position_analytical, composite_score,
    save_results, FEATURE_DIM, DATA_DIR, RESULTS_DIR,
)
from train_eval_mlp import RelayMLP, load_training_data, train_mlp, evaluate_mlp


def main():
    # Load only 2,000 samples (same as VLA)
    X, y = load_training_data(max_samples=2000)

    # 90/10 split (1800 train, 200 val — same as VLA)
    n_train = int(0.9 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train
    print("\nTraining MLP on 2,000 samples (fair comparison with VLA)...")
    t0 = time.time()
    model = train_mlp(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # Evaluate on canonical 100 scenarios
    print("\nEvaluating on 100 canonical scenarios...")
    scenarios = generate_scenarios(100)
    summary, per_scenario = evaluate_mlp(model, scenarios)

    # Analytical baseline
    analytical_tp = []
    for s in scenarios:
        pos = get_position_analytical(s)
        m = compute_channel_metrics(pos, s)
        analytical_tp.append(m['total_throughput'])

    results = {
        'method': 'mlp_2k',
        'architecture': '37 -> 256 -> BN -> 128 -> BN -> 64 -> 3',
        'training_samples': 2000,
        'training_time_s': train_time,
        'summary': summary,
        'per_scenario': per_scenario,
        'comparison': {
            'analytical_mean_throughput': float(np.mean(analytical_tp)),
            'mlp2k_vs_analytical_pct': float(
                (summary['throughput_mbps']['mean'] - np.mean(analytical_tp))
                / np.mean(analytical_tp) * 100
            ),
        },
    }

    print(f"\n{'='*60}")
    print("MLP-2K BASELINE RESULTS (Fair Comparison)")
    print(f"{'='*60}")
    print(f"  Throughput: {summary['throughput_mbps']['mean']:.1f} +/- "
          f"{summary['throughput_mbps']['ci95']:.1f} Mbps")
    print(f"  Fairness:   {summary['fairness_mean']:.3f}")
    print(f"  Coverage:   {summary['coverage_mean']*100:.1f}%")
    print(f"  Latency:    {summary['latency_ms']['mean']:.3f} ms")
    print(f"  vs Analytical: {results['comparison']['mlp2k_vs_analytical_pct']:+.1f}%")

    path = save_results(results, 'mlp_2k_baseline')
    print(f"\nDone! Results at {path}")

    # Save model
    model_path = os.path.join(RESULTS_DIR, 'mlp_2k_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
