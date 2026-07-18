#!/usr/bin/env python3
"""
Obstacle-Aware MLP Baseline — Train & Evaluate

This script creates a FAIR comparison for the VLA extensibility claim:
- MLP-Obs receives the same obstacle information as VLA-Obs
- Input dimension: 40 (37 standard + 3 binary obstacle sector flags)
- Training: 500 obstacle-aware samples (same as VLA-Obs fine-tuning)
- Evaluation: 50 obstacle scenarios

This addresses Reviewer 1 Point 3 and Reviewer 2 Point 2:
"The MLP baseline is blind to obstacles while VLA sees them — unfair comparison"
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
    extract_features_obstacle, get_position_analytical,
    composite_score, save_results, FEATURE_DIM_OBSTACLE,
    DATA_DIR, RESULTS_DIR, ObstacleScenario, Obstacle, Scenario,
)


# ---------------------------------------------------------------------------
# Obstacle-Aware MLP Architecture (40-dim input)
# ---------------------------------------------------------------------------

class RelayMLPObstacle(nn.Module):
    """MLP with obstacle-aware input (40 dimensions)."""

    def __init__(self, input_dim=FEATURE_DIM_OBSTACLE):
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
# Generate obstacle training data with DE optimization
# ---------------------------------------------------------------------------

def generate_obstacle_training_data(num_samples: int = 500, seed: int = 123):
    """Generate training data for obstacle scenarios using differential evolution.

    For each scenario:
    1. Generate random obstacle configuration
    2. Run DE optimizer to find optimal UAV position considering obstacles
    3. Extract 40-dim features and optimal position target
    """
    from scipy.optimize import differential_evolution

    random.seed(seed)
    np.random.seed(seed)

    features_list = []
    targets_list = []
    scenarios_data = []

    print(f"Generating {num_samples} obstacle-aware training samples...")

    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

        # Generate a random obstacle scenario
        scenario = generate_obstacle_scenarios(
            num_scenarios=1,
            seed=seed + i,
            num_obstacles_range=(1, 3)
        )[0]
        scenario.id = i

        # Define objective for DE: maximize throughput considering obstacles
        def objective(pos_array):
            uav_pos = np.array([
                np.clip(pos_array[0], 0, 100),
                np.clip(pos_array[1], 0, 100),
                np.clip(pos_array[2], 10, 40),
            ])
            metrics = compute_channel_metrics_obstacles(uav_pos, scenario)
            # Negative because DE minimizes
            return -composite_score(metrics)

        # Run DE optimizer
        bounds = [(0, 100), (0, 100), (10, 40)]
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            seed=seed + i,
            tol=0.01,
            polish=False,
        )

        optimal_pos = np.array([
            np.clip(result.x[0], 0, 100),
            np.clip(result.x[1], 0, 100),
            np.clip(result.x[2], 10, 40),
        ])

        # Extract features
        features = extract_features_obstacle(scenario)
        features_list.append(features)
        targets_list.append(optimal_pos.astype(np.float32))

        # Store scenario metadata
        scenarios_data.append({
            'id': i,
            'num_users': scenario.num_users,
            'num_obstacles': len(scenario.obstacles),
            'optimal_position': optimal_pos.tolist(),
            'optimal_throughput': -result.fun,  # Negate back
        })

    X = np.stack(features_list)
    y = np.stack(targets_list)

    print(f"Generated training data: X={X.shape}, y={y.shape}")
    return X, y, scenarios_data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp_obstacle(X_train, y_train, X_val, y_val,
                       epochs=200, batch_size=32, lr=1e-3):
    """Train the obstacle-aware MLP and return the best model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    model = RelayMLPObstacle().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float('inf')
    best_state = None
    train_losses = []
    val_losses = []

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

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"Best validation loss: {best_val_loss:.4f}")
    return model, {'train_losses': train_losses, 'val_losses': val_losses}


# ---------------------------------------------------------------------------
# Evaluation on obstacle scenarios
# ---------------------------------------------------------------------------

def evaluate_mlp_obstacle(model, scenarios):
    """Evaluate obstacle-aware MLP on obstacle scenarios."""
    device = next(model.parameters()).device
    model.eval()

    results_per_scenario = []

    for scenario in scenarios:
        feat = extract_features_obstacle(scenario)
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
            'bs_uav_blocked': metrics['bs_uav_blocked'],
            'num_blocked_users': metrics['num_blocked_users'],
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
            'min': float(np.min(tp_arr)),
            'max': float(np.max(tp_arr)),
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


def evaluate_blind_mlp_obstacle(scenarios):
    """Evaluate the BLIND MLP (no obstacle awareness) on obstacle scenarios.

    This simulates what happens when the standard MLP is deployed in
    environments with obstacles it wasn't trained on.
    """
    from train_eval_mlp import RelayMLP
    from eval_common import extract_features, Scenario, FEATURE_DIM

    # Load the pre-trained blind MLP
    model_path = os.path.join(RESULTS_DIR, 'mlp_2k_model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(RESULTS_DIR, 'mlp_model.pt')

    if not os.path.exists(model_path):
        print("WARNING: No pre-trained MLP found. Skipping blind MLP comparison.")
        return None, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RelayMLP(input_dim=FEATURE_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results_per_scenario = []

    for obs_scenario in scenarios:
        # Convert to regular Scenario for feature extraction
        base_scenario = Scenario(
            id=obs_scenario.id,
            num_users=obs_scenario.num_users,
            user_positions=obs_scenario.user_positions,
            user_requirements=obs_scenario.user_requirements,
            bs_position=obs_scenario.bs_position,
            initial_uav_position=obs_scenario.initial_uav_position,
        )

        feat = extract_features(base_scenario)
        feat_t = torch.tensor(feat).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(feat_t).cpu().numpy()[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        pos = np.array([
            np.clip(pred[0], 0, 100),
            np.clip(pred[1], 0, 100),
            np.clip(pred[2], 10, 40),
        ])

        # Evaluate with obstacle channel model (MLP doesn't know about obstacles)
        metrics = compute_channel_metrics_obstacles(pos, obs_scenario)

        results_per_scenario.append({
            'scenario_id': obs_scenario.id,
            'predicted_position': pos.tolist(),
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage_rate': metrics['coverage_rate'],
            'inference_ms': inference_ms,
            'composite_score': composite_score(metrics),
            'num_obstacles': len(obs_scenario.obstacles),
            'bs_uav_blocked': metrics['bs_uav_blocked'],
            'num_blocked_users': metrics['num_blocked_users'],
        })

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("OBSTACLE-AWARE MLP BASELINE")
    print("Fair comparison: MLP-Obs vs VLA-Obs (both see obstacle info)")
    print("=" * 70)

    # Generate training data (500 samples, same as VLA-Obs fine-tuning)
    X, y, train_metadata = generate_obstacle_training_data(num_samples=500, seed=123)

    # 90/10 split (450 train, 50 val)
    n_train = int(0.9 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train
    print("\nTraining obstacle-aware MLP on 500 samples...")
    t0 = time.time()
    model, learning_curves = train_mlp_obstacle(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # Generate test scenarios (50 obstacle scenarios, different seed)
    print("\nGenerating 50 obstacle test scenarios...")
    test_scenarios = generate_obstacle_scenarios(num_scenarios=50, seed=999)

    # Evaluate obstacle-aware MLP
    print("Evaluating MLP-Obs...")
    summary_obs, per_scenario_obs = evaluate_mlp_obstacle(model, test_scenarios)

    # Evaluate blind MLP for comparison
    print("Evaluating blind MLP (no obstacle info)...")
    summary_blind, per_scenario_blind = evaluate_blind_mlp_obstacle(test_scenarios)

    # Compute analytical baseline on obstacle scenarios
    print("Computing analytical baseline...")
    analytical_tp = []
    for s in test_scenarios:
        base_scenario = Scenario(
            id=s.id,
            num_users=s.num_users,
            user_positions=s.user_positions,
            user_requirements=s.user_requirements,
            bs_position=s.bs_position,
            initial_uav_position=s.initial_uav_position,
        )
        pos = get_position_analytical(base_scenario)
        # Evaluate analytical position with obstacle channel model
        metrics = compute_channel_metrics_obstacles(pos, s)
        analytical_tp.append(metrics['total_throughput'])

    # Compile results
    results = {
        'method': 'mlp_obstacle',
        'architecture': '40 -> 256 -> BN -> 128 -> BN -> 64 -> 3',
        'training_samples': 500,
        'training_time_s': train_time,
        'test_scenarios': 50,
        'summary_mlp_obs': summary_obs,
        'summary_mlp_blind': summary_blind,
        'per_scenario_mlp_obs': per_scenario_obs,
        'per_scenario_mlp_blind': per_scenario_blind,
        'learning_curves': learning_curves,
        'comparison': {
            'analytical_mean_throughput': float(np.mean(analytical_tp)),
            'mlp_obs_mean_throughput': summary_obs['throughput_mbps']['mean'],
            'mlp_blind_mean_throughput': summary_blind['throughput_mbps']['mean'] if summary_blind else None,
            'mlp_obs_vs_blind_pct': float(
                (summary_obs['throughput_mbps']['mean'] - summary_blind['throughput_mbps']['mean'])
                / summary_blind['throughput_mbps']['mean'] * 100
            ) if summary_blind else None,
            'mlp_obs_vs_analytical_pct': float(
                (summary_obs['throughput_mbps']['mean'] - np.mean(analytical_tp))
                / np.mean(analytical_tp) * 100
            ),
        },
    }

    # Print results
    print(f"\n{'='*70}")
    print("OBSTACLE-AWARE MLP RESULTS (for Table VII)")
    print(f"{'='*70}")
    print(f"\n  MLP-Obs (obstacle-aware, 500 samples):")
    print(f"    Throughput: {summary_obs['throughput_mbps']['mean']:.1f} +/- "
          f"{summary_obs['throughput_mbps']['ci95']:.1f} Mbps")
    print(f"    Fairness:   {summary_obs['fairness_mean']:.3f}")
    print(f"    Coverage:   {summary_obs['coverage_mean']*100:.1f}%")
    print(f"    Latency:    {summary_obs['latency_ms']['mean']:.3f} ms")

    if summary_blind:
        print(f"\n  MLP-Blind (no obstacle info):")
        print(f"    Throughput: {summary_blind['throughput_mbps']['mean']:.1f} +/- "
              f"{summary_blind['throughput_mbps']['ci95']:.1f} Mbps")
        print(f"    Fairness:   {summary_blind['fairness_mean']:.3f}")
        print(f"    Coverage:   {summary_blind['coverage_mean']*100:.1f}%")

    print(f"\n  Analytical baseline (on obstacle scenarios):")
    print(f"    Throughput: {np.mean(analytical_tp):.1f} Mbps")

    print(f"\n  Improvement over blind MLP: "
          f"{results['comparison']['mlp_obs_vs_blind_pct']:+.1f}%")
    print(f"  Improvement over analytical: "
          f"{results['comparison']['mlp_obs_vs_analytical_pct']:+.1f}%")

    # Save results
    path = save_results(results, 'mlp_obstacle_baseline')
    print(f"\nResults saved to {path}")

    # Save model
    model_path = os.path.join(RESULTS_DIR, 'mlp_obstacle_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Print Table VII row (for paper)
    print(f"\n{'='*70}")
    print("TABLE VII ROW (add to paper):")
    print(f"{'='*70}")
    print(f"MLP-Obs (500)  |  {summary_obs['throughput_mbps']['mean']:.1f}  |  "
          f"{summary_obs['fairness_mean']:.3f}  |  {summary_obs['coverage_mean']*100:.1f}%  |  "
          f"{summary_obs['latency_ms']['mean']:.2f} ms")


if __name__ == '__main__':
    main()
