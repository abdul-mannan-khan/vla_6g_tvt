#!/usr/bin/env python3
"""
M2: MLP Baseline — Train & Evaluate

Architecture: 37 -> 256 -> ReLU -> BN -> 128 -> ReLU -> BN -> 64 -> ReLU -> 3
Training: MSE loss, 200 epochs, batch=64, lr=1e-3, on 8000 training samples
Evaluate: Same 100 scenarios, compute channel metrics at predicted position
"""

import sys
import os
import json
import time
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


# ---------------------------------------------------------------------------
# MLP Architecture
# ---------------------------------------------------------------------------

class RelayMLP(nn.Module):
    """Simple MLP for UAV relay position regression."""

    def __init__(self, input_dim=FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # x, y, z
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(max_samples=8000):
    """Load training samples and extract features + targets."""
    data_path = os.path.join(DATA_DIR, 'vla_training_data_20260130_202752.json')
    print(f"Loading training data from {data_path}...")
    with open(data_path) as f:
        raw = json.load(f)

    samples = raw['samples'][:max_samples]
    print(f"Using {len(samples)} training samples")

    features_list = []
    targets_list = []

    for s in samples:
        user_positions = [np.array(p) for p in s['user_positions']]
        feat = extract_features_from_raw(
            bs_position=np.array(s['bs_position']),
            uav_position=np.array(s['uav_position']),
            user_positions=user_positions,
            user_requirements=[25.0] * s['num_users'],  # default QoS
            user_snrs=s['current_user_snrs'],
            user_rates=s['current_user_rates'],
            throughput=s['current_throughput'],
            fairness=s['current_fairness'],
        )
        features_list.append(feat)
        targets_list.append(np.array(s['optimal_position'], dtype=np.float32))

    X = np.stack(features_list)
    y = np.stack(targets_list)
    print(f"Features shape: {X.shape}, Targets shape: {y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp(X_train, y_train, X_val, y_val,
              epochs=200, batch_size=64, lr=1e-3):
    """Train the MLP and return the best model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    model = RelayMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"Best validation loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_mlp(model, scenarios):
    """Evaluate trained MLP on the 100 canonical scenarios."""
    device = next(model.parameters()).device
    model.eval()

    results_per_scenario = []

    for scenario in scenarios:
        feat = extract_features(scenario)
        feat_t = torch.tensor(feat).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(feat_t).cpu().numpy()[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        # Clip to valid range
        pos = np.array([
            np.clip(pred[0], 0, 100),
            np.clip(pred[1], 0, 100),
            np.clip(pred[2], 10, 40),
        ])

        metrics = compute_channel_metrics(pos, scenario)

        results_per_scenario.append({
            'scenario_id': scenario.id,
            'predicted_position': pos.tolist(),
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage_rate': metrics['coverage_rate'],
            'inference_ms': inference_ms,
            'composite_score': composite_score(metrics),
        })

    # Aggregate
    tp_arr = np.array([r['throughput'] for r in results_per_scenario])
    fair_arr = np.array([r['fairness'] for r in results_per_scenario])
    cov_arr = np.array([r['coverage_rate'] for r in results_per_scenario])
    lat_arr = np.array([r['inference_ms'] for r in results_per_scenario])

    summary = {
        'throughput_mbps': {
            'mean': float(np.mean(tp_arr)),
            'std': float(np.std(tp_arr)),
            'ci95': float(1.96 * np.std(tp_arr, ddof=1) / np.sqrt(len(tp_arr))),
        },
        'fairness_mean': float(np.mean(fair_arr)),
        'coverage_mean': float(np.mean(cov_arr)),
        'latency_ms': {
            'mean': float(np.mean(lat_arr)),
            'p50': float(np.percentile(lat_arr, 50)),
            'p95': float(np.percentile(lat_arr, 95)),
        },
    }

    return summary, results_per_scenario


def main():
    # Load data
    X, y = load_training_data(max_samples=8000)

    # 90/10 split
    n_train = int(0.9 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train
    print("\nTraining MLP...")
    t0 = time.time()
    model = train_mlp(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # Evaluate on canonical 100 scenarios
    print("\nEvaluating on 100 canonical scenarios...")
    scenarios = generate_scenarios(100)
    summary, per_scenario = evaluate_mlp(model, scenarios)

    # Also get analytical baseline for comparison
    analytical_tp = []
    for s in scenarios:
        pos = get_position_analytical(s)
        m = compute_channel_metrics(pos, s)
        analytical_tp.append(m['total_throughput'])

    results = {
        'method': 'mlp',
        'architecture': '37 -> 256 -> BN -> 128 -> BN -> 64 -> 3',
        'training_samples': 8000,
        'training_time_s': train_time,
        'summary': summary,
        'per_scenario': per_scenario,
        'comparison': {
            'analytical_mean_throughput': float(np.mean(analytical_tp)),
            'mlp_vs_analytical_pct': float(
                (summary['throughput_mbps']['mean'] - np.mean(analytical_tp))
                / np.mean(analytical_tp) * 100
            ),
        },
    }

    print(f"\n{'='*60}")
    print("MLP BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"  Throughput: {summary['throughput_mbps']['mean']:.1f} +/- "
          f"{summary['throughput_mbps']['ci95']:.1f} Mbps")
    print(f"  Fairness:   {summary['fairness_mean']:.3f}")
    print(f"  Coverage:   {summary['coverage_mean']*100:.1f}%")
    print(f"  Latency:    {summary['latency_ms']['mean']:.3f} ms "
          f"(p50={summary['latency_ms']['p50']:.3f})")
    print(f"  vs Analytical: {results['comparison']['mlp_vs_analytical_pct']:+.1f}%")

    path = save_results(results, 'mlp_baseline')
    print(f"\nDone! Results at {path}")

    # Save model weights
    model_path = os.path.join(RESULTS_DIR, 'mlp_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
