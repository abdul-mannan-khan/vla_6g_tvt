#!/usr/bin/env python3
"""
Fair Obstacle-Aware MLP Comparison

This script provides a FAIR comparison between MLP and VLA for obstacle scenarios:
1. Trains MLP-Obs on 2K obstacle samples (same data budget as VLA training)
2. Uses rich obstacle features (51-dim) for proper information parity
3. Compares against blind MLP and VLA baselines

Key insight: Both paradigms should perform similarly when given equivalent
information and training data. The VLA's advantage is interface flexibility,
not inherent performance superiority.
"""

import sys
import os
import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (
    generate_obstacle_scenarios, compute_channel_metrics_obstacles,
    extract_features_obstacle_rich, get_position_analytical,
    composite_score, save_results, FEATURE_DIM_OBSTACLE_RICH,
    DATA_DIR, RESULTS_DIR, ObstacleScenario, Obstacle, Scenario,
    check_los_blocked,
)


# ---------------------------------------------------------------------------
# Rich Obstacle-Aware MLP Architecture (51-dim input)
# ---------------------------------------------------------------------------

class RelayMLPObstacleRich(nn.Module):
    """MLP with rich obstacle-aware input (51 dimensions)."""

    def __init__(self, input_dim=FEATURE_DIM_OBSTACLE_RICH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # x, y, z
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Generate obstacle training data with DE optimization
# ---------------------------------------------------------------------------

def generate_obstacle_training_data(num_samples: int = 2000, seed: int = 42):
    """Generate training data for obstacle scenarios using differential evolution.

    Uses 2K samples for fair comparison with VLA (which was trained on 2K samples).
    """
    from scipy.optimize import differential_evolution

    random.seed(seed)
    np.random.seed(seed)

    features_list = []
    targets_list = []
    scenarios_data = []

    print(f"Generating {num_samples} obstacle-aware training samples...")

    for i in range(num_samples):
        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

        # Generate a random obstacle scenario
        scenario = generate_obstacle_scenarios(
            num_scenarios=1,
            seed=seed + i * 7,  # Different seed multiplier for variety
            num_obstacles_range=(1, 3)
        )[0]
        scenario.id = i

        # Define objective for DE: maximize composite score considering obstacles
        def objective(pos_array):
            uav_pos = np.array([
                np.clip(pos_array[0], 0, 100),
                np.clip(pos_array[1], 0, 100),
                np.clip(pos_array[2], 10, 40),
            ])
            metrics = compute_channel_metrics_obstacles(uav_pos, scenario)
            # Negative because DE minimizes
            return -composite_score(metrics)

        # Run DE optimizer with more iterations for better solutions
        bounds = [(0, 100), (0, 100), (10, 40)]
        result = differential_evolution(
            objective,
            bounds,
            maxiter=150,
            popsize=20,
            seed=seed + i,
            tol=0.001,
            polish=True,
        )

        optimal_pos = np.array([
            np.clip(result.x[0], 0, 100),
            np.clip(result.x[1], 0, 100),
            np.clip(result.x[2], 10, 40),
        ])

        # Extract rich features
        features = extract_features_obstacle_rich(scenario)
        features_list.append(features)
        targets_list.append(optimal_pos.astype(np.float32))

        # Store scenario metadata
        optimal_metrics = compute_channel_metrics_obstacles(optimal_pos, scenario)
        scenarios_data.append({
            'id': i,
            'num_users': scenario.num_users,
            'num_obstacles': len(scenario.obstacles),
            'optimal_position': optimal_pos.tolist(),
            'optimal_throughput': optimal_metrics['total_throughput'],
        })

    X = np.stack(features_list)
    y = np.stack(targets_list)

    print(f"Generated training data: X={X.shape}, y={y.shape}")
    return X, y, scenarios_data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp_obstacle_rich(X_train, y_train, X_val, y_val,
                            epochs=300, batch_size=32, lr=1e-3):
    """Train the rich obstacle-aware MLP and return the best model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    model = RelayMLPObstacleRich().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float('inf')
    best_state = None
    train_losses = []
    val_losses = []
    patience = 30
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)
        train_losses.append(train_loss)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"Best validation loss: {best_val_loss:.4f}")
    return model, {'train_losses': train_losses, 'val_losses': val_losses}


# ---------------------------------------------------------------------------
# Evaluation on obstacle scenarios
# ---------------------------------------------------------------------------

def evaluate_mlp_obstacle_rich(model, scenarios):
    """Evaluate rich obstacle-aware MLP on obstacle scenarios."""
    device = next(model.parameters()).device
    model.eval()

    results_per_scenario = []

    for scenario in scenarios:
        feat = extract_features_obstacle_rich(scenario)
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

        metrics = compute_channel_metrics_obstacles(pos, scenario)

        results_per_scenario.append({
            'scenario_id': scenario.id,
            'predicted_position': pos.tolist(),
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage_rate': metrics['coverage_rate'],
            'inference_ms': inference_ms,
            'composite_score': composite_score(metrics),
            'num_obstacles': len(scenario.obstacles),
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


def evaluate_de_optimizer(scenarios):
    """Run DE optimizer on each scenario (upper bound)."""
    from scipy.optimize import differential_evolution

    results = []
    print("  Running DE optimizer (upper bound)...")

    for i, scenario in enumerate(scenarios):
        if (i + 1) % 25 == 0:
            print(f"    Optimizing scenario {i + 1}/{len(scenarios)}")

        def objective(pos_array):
            uav_pos = np.array([
                np.clip(pos_array[0], 0, 100),
                np.clip(pos_array[1], 0, 100),
                np.clip(pos_array[2], 10, 40),
            ])
            metrics = compute_channel_metrics_obstacles(uav_pos, scenario)
            return -composite_score(metrics)

        t0 = time.perf_counter()
        result = differential_evolution(
            objective,
            bounds=[(0, 100), (0, 100), (10, 40)],
            maxiter=200,
            popsize=20,
            seed=42 + i,
            tol=0.001,
            polish=True,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        pos = np.array([
            np.clip(result.x[0], 0, 100),
            np.clip(result.x[1], 0, 100),
            np.clip(result.x[2], 10, 40),
        ])
        metrics = compute_channel_metrics_obstacles(pos, scenario)

        results.append({
            'scenario_id': scenario.id,
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage_rate': metrics['coverage_rate'],
            'inference_ms': inference_ms,
        })

    tp_arr = np.array([r['throughput'] for r in results])
    fair_arr = np.array([r['fairness'] for r in results])
    cov_arr = np.array([r['coverage_rate'] for r in results])

    return {
        'throughput_mbps': {
            'mean': float(np.mean(tp_arr)),
            'std': float(np.std(tp_arr)),
            'ci95': float(1.96 * np.std(tp_arr, ddof=1) / np.sqrt(len(tp_arr))),
        },
        'fairness_mean': float(np.mean(fair_arr)),
        'coverage_mean': float(np.mean(cov_arr)),
    }


def evaluate_analytical(scenarios):
    """Evaluate analytical baseline on obstacle scenarios."""
    results = []

    for scenario in scenarios:
        base_scenario = Scenario(
            id=scenario.id,
            num_users=scenario.num_users,
            user_positions=scenario.user_positions,
            user_requirements=scenario.user_requirements,
            bs_position=scenario.bs_position,
            initial_uav_position=scenario.initial_uav_position,
        )

        t0 = time.perf_counter()
        pos = get_position_analytical(base_scenario)
        inference_ms = (time.perf_counter() - t0) * 1000

        metrics = compute_channel_metrics_obstacles(pos, scenario)

        results.append({
            'scenario_id': scenario.id,
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage_rate': metrics['coverage_rate'],
            'inference_ms': inference_ms,
        })

    tp_arr = np.array([r['throughput'] for r in results])
    fair_arr = np.array([r['fairness'] for r in results])
    cov_arr = np.array([r['coverage_rate'] for r in results])

    return {
        'throughput_mbps': {
            'mean': float(np.mean(tp_arr)),
            'std': float(np.std(tp_arr)),
            'ci95': float(1.96 * np.std(tp_arr, ddof=1) / np.sqrt(len(tp_arr))),
        },
        'fairness_mean': float(np.mean(fair_arr)),
        'coverage_mean': float(np.mean(cov_arr)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FAIR OBSTACLE-AWARE MLP COMPARISON")
    print("Training on 2K samples with rich features (51-dim)")
    print("=" * 70)

    # Generate training data (2K samples, same as VLA)
    X, y, train_metadata = generate_obstacle_training_data(num_samples=2000, seed=42)

    # 90/10 split (1800 train, 200 val)
    n_train = int(0.9 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train
    print("\nTraining MLP-Obs-Rich on 2K samples...")
    t0 = time.time()
    model, learning_curves = train_mlp_obstacle_rich(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # Generate test scenarios (100 obstacle scenarios for robust evaluation)
    print("\nGenerating 100 obstacle test scenarios...")
    test_scenarios = generate_obstacle_scenarios(num_scenarios=100, seed=9999)

    # Evaluate methods
    print("\nEvaluating methods:")

    print("  MLP-Obs-Rich (2K, 51-dim)...")
    summary_mlp_rich, per_scenario_mlp = evaluate_mlp_obstacle_rich(model, test_scenarios)

    print("  Analytical baseline...")
    summary_analytical = evaluate_analytical(test_scenarios)

    print("  DE Optimizer (upper bound)...")
    summary_de = evaluate_de_optimizer(test_scenarios)

    # Calculate recovery ratio
    recovery_ratio = (summary_mlp_rich['throughput_mbps']['mean'] - summary_analytical['throughput_mbps']['mean']) / \
                     (summary_de['throughput_mbps']['mean'] - summary_analytical['throughput_mbps']['mean'])

    # Compile results
    results = {
        'method': 'mlp_obstacle_rich',
        'architecture': '51 -> 256 -> BN -> 128 -> BN -> 64 -> 3',
        'training_samples': 2000,
        'training_time_s': train_time,
        'test_scenarios': 100,
        'summary_mlp_rich': summary_mlp_rich,
        'summary_analytical': summary_analytical,
        'summary_de': summary_de,
        'learning_curves': learning_curves,
        'comparison': {
            'analytical_throughput': summary_analytical['throughput_mbps']['mean'],
            'mlp_rich_throughput': summary_mlp_rich['throughput_mbps']['mean'],
            'de_throughput': summary_de['throughput_mbps']['mean'],
            'mlp_vs_analytical_pct': float(
                (summary_mlp_rich['throughput_mbps']['mean'] - summary_analytical['throughput_mbps']['mean'])
                / summary_analytical['throughput_mbps']['mean'] * 100
            ),
            'recovery_ratio': float(recovery_ratio),
        },
    }

    # Print results
    print(f"\n{'='*70}")
    print("MLP-Obs-Rich RESULTS (2K samples, 51-dim features)")
    print(f"{'='*70}")
    print(f"\n  MLP-Obs-Rich:")
    print(f"    Throughput: {summary_mlp_rich['throughput_mbps']['mean']:.1f} +/- "
          f"{summary_mlp_rich['throughput_mbps']['ci95']:.1f} Mbps")
    print(f"    Fairness:   {summary_mlp_rich['fairness_mean']:.3f}")
    print(f"    Coverage:   {summary_mlp_rich['coverage_mean']*100:.1f}%")
    print(f"    Latency:    {summary_mlp_rich['latency_ms']['mean']:.3f} ms")

    print(f"\n  Analytical baseline:")
    print(f"    Throughput: {summary_analytical['throughput_mbps']['mean']:.1f} Mbps")

    print(f"\n  DE Optimizer (upper bound):")
    print(f"    Throughput: {summary_de['throughput_mbps']['mean']:.1f} Mbps")

    print(f"\n  Performance:")
    print(f"    vs Analytical: {results['comparison']['mlp_vs_analytical_pct']:+.1f}%")
    print(f"    Recovery ratio: {recovery_ratio:.1%}")

    # Save results
    path = save_results(results, 'mlp_obstacle_rich_fair')
    print(f"\nResults saved to {path}")

    # Save model
    model_path = os.path.join(RESULTS_DIR, 'mlp_obstacle_rich_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Print Table VII row
    print(f"\n{'='*70}")
    print("TABLE VII ROW:")
    print(f"{'='*70}")
    print(f"MLP-Obs-Rich (2K)  |  {summary_mlp_rich['throughput_mbps']['mean']:.1f}  |  "
          f"{summary_mlp_rich['fairness_mean']:.3f}  |  {summary_mlp_rich['coverage_mean']*100:.1f}%  |  "
          f"{summary_mlp_rich['latency_ms']['mean']:.2f} ms")


if __name__ == '__main__':
    main()
