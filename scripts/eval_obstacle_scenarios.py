#!/usr/bin/env python3
"""
Obstacle Scenario Evaluation — Compare all methods on obstacle scenarios

This script generates Table VII data comparing:
- MLP-Obs (obstacle-aware MLP, 500 samples) -- NEW FAIR COMPARISON
- MLP-Blind (standard MLP, no obstacle info)
- VLA-Obs (VLA fine-tuned on obstacles, 500 samples)
- VLA-Blind (standard VLA, no obstacle info)
- Analytical baseline
- DE Optimizer (upper bound)

Addresses reviewer concerns about unfair obstacle comparison.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (
    generate_obstacle_scenarios, compute_channel_metrics_obstacles,
    extract_features_obstacle, extract_features, get_position_analytical,
    composite_score, save_results, FEATURE_DIM, FEATURE_DIM_OBSTACLE,
    RESULTS_DIR, ObstacleScenario, Obstacle, Scenario,
    format_vla_prompt_obstacles,
)

import torch


def evaluate_de_optimizer(scenarios):
    """Run DE optimizer on each scenario (upper bound)."""
    from scipy.optimize import differential_evolution

    results = []
    print("  Running DE optimizer (upper bound)...")

    for i, scenario in enumerate(scenarios):
        if (i + 1) % 10 == 0:
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

    return aggregate_results(results)


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

    return aggregate_results(results)


def evaluate_mlp_blind(scenarios):
    """Evaluate standard MLP (37-dim, no obstacle info)."""
    from train_eval_mlp import RelayMLP

    model_path = os.path.join(RESULTS_DIR, 'mlp_2k_model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(RESULTS_DIR, 'mlp_model.pt')

    if not os.path.exists(model_path):
        print("  WARNING: No pre-trained MLP found. Skipping MLP-Blind.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RelayMLP(input_dim=FEATURE_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

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

        metrics = compute_channel_metrics_obstacles(pos, scenario)

        results.append({
            'scenario_id': scenario.id,
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage_rate': metrics['coverage_rate'],
            'inference_ms': inference_ms,
        })

    return aggregate_results(results)


def evaluate_mlp_obs(scenarios):
    """Evaluate obstacle-aware MLP (40-dim)."""
    from train_eval_mlp_obstacle import RelayMLPObstacle

    model_path = os.path.join(RESULTS_DIR, 'mlp_obstacle_model.pt')

    if not os.path.exists(model_path):
        print("  WARNING: No obstacle-aware MLP found. Run train_eval_mlp_obstacle.py first.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RelayMLPObstacle(input_dim=FEATURE_DIM_OBSTACLE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results = []

    for scenario in scenarios:
        feat = extract_features_obstacle(scenario)
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

        metrics = compute_channel_metrics_obstacles(pos, scenario)

        results.append({
            'scenario_id': scenario.id,
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage_rate': metrics['coverage_rate'],
            'inference_ms': inference_ms,
        })

    return aggregate_results(results)


def aggregate_results(results):
    """Compute summary statistics from per-scenario results."""
    if not results:
        return None

    tp_arr = np.array([r['throughput'] for r in results])
    fair_arr = np.array([r['fairness'] for r in results])
    cov_arr = np.array([r['coverage_rate'] for r in results])
    lat_arr = np.array([r['inference_ms'] for r in results])

    return {
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
        'per_scenario': results,
    }


def print_table_row(name, summary, ref_throughput=None):
    """Print a formatted table row."""
    if summary is None:
        print(f"  {name:20s}  |  N/A")
        return

    tp = summary['throughput_mbps']['mean']
    tp_ci = summary['throughput_mbps']['ci95']
    fair = summary['fairness_mean']
    cov = summary['coverage_mean'] * 100
    lat = summary['latency_ms']['mean']

    delta_str = ""
    if ref_throughput is not None:
        delta = (tp - ref_throughput) / ref_throughput * 100
        delta_str = f"  ({delta:+.1f}%)"

    print(f"  {name:20s}  |  {tp:6.1f} +/- {tp_ci:4.1f}  |  "
          f"{fair:.3f}  |  {cov:5.1f}%  |  {lat:8.2f} ms{delta_str}")


def main():
    print("=" * 80)
    print("OBSTACLE SCENARIO EVALUATION")
    print("Generating Table VII data for fair obstacle comparison")
    print("=" * 80)

    # Generate 50 test scenarios with obstacles
    print("\nGenerating 50 obstacle test scenarios (seed=999)...")
    scenarios = generate_obstacle_scenarios(num_scenarios=50, seed=999)

    # Evaluate all methods
    print("\nEvaluating methods:")

    print("  Analytical baseline...")
    summary_analytical = evaluate_analytical(scenarios)

    print("  MLP-Blind (37-dim, no obstacle info)...")
    summary_mlp_blind = evaluate_mlp_blind(scenarios)

    print("  MLP-Obs (40-dim, obstacle-aware)...")
    summary_mlp_obs = evaluate_mlp_obs(scenarios)

    print("  DE Optimizer (upper bound)...")
    summary_de = evaluate_de_optimizer(scenarios)

    # Compile results
    results = {
        'test_scenarios': 50,
        'methods': {
            'analytical': summary_analytical,
            'mlp_blind': summary_mlp_blind,
            'mlp_obs': summary_mlp_obs,
            'de_optimizer': summary_de,
        },
    }

    # Print table
    ref_tp = summary_analytical['throughput_mbps']['mean']

    print(f"\n{'='*80}")
    print("TABLE VII: Obstacle Scenario Performance (50 scenarios)")
    print(f"{'='*80}")
    print(f"  {'Method':20s}  |  {'Throughput':14s}  |  {'Fair':5s}  |  "
          f"{'Cov':6s}  |  {'Latency':10s}")
    print(f"  {'-'*20}  |  {'-'*14}  |  {'-'*5}  |  {'-'*6}  |  {'-'*10}")

    print_table_row("Analytical", summary_analytical)
    print_table_row("MLP-Blind", summary_mlp_blind, ref_tp)
    print_table_row("MLP-Obs (500)", summary_mlp_obs, ref_tp)
    print_table_row("DE Optimizer", summary_de, ref_tp)

    # Key comparison
    if summary_mlp_obs and summary_mlp_blind:
        mlp_obs_tp = summary_mlp_obs['throughput_mbps']['mean']
        mlp_blind_tp = summary_mlp_blind['throughput_mbps']['mean']
        improvement = (mlp_obs_tp - mlp_blind_tp) / mlp_blind_tp * 100

        print(f"\n{'='*80}")
        print("KEY FINDING:")
        print(f"{'='*80}")
        print(f"  MLP-Obs improvement over MLP-Blind: {improvement:+.1f}%")
        print(f"  MLP-Obs throughput: {mlp_obs_tp:.1f} Mbps")
        print(f"  MLP-Blind throughput: {mlp_blind_tp:.1f} Mbps")

        if improvement > 3:
            print("\n  CONCLUSION: Obstacle-awareness provides significant benefit.")
            print("  The VLA extensibility claim holds for MLP as well.")
        else:
            print("\n  CONCLUSION: Obstacle-awareness provides marginal benefit for MLP.")
            print("  Consider whether VLA's prompt-based interface offers unique advantages.")

    # Save results
    path = save_results(results, 'obstacle_scenario_comparison')
    print(f"\nResults saved to: {path}")


if __name__ == '__main__':
    main()
