#!/usr/bin/env python3
"""
Train SAC (Soft Actor-Critic) for 500K timesteps (5x the previous 100K budget).

Architecture: MlpPolicy [256, 128]
Parallel envs: 8 x DummyVecEnv
Hyperparameters: lr=3e-4, buffer=100K, batch=256, tau=0.005, gamma=0.99
                 train_freq=8, gradient_steps=1, learning_starts=1000
Evaluation: 100 canonical scenarios (seed=42)
"""

import sys
import os
import json
import time
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (
    generate_scenarios, compute_channel_metrics, composite_score,
    extract_features, get_position_analytical,
    save_results, FEATURE_DIM, DATA_DIR,
)
from train_eval_drl import UAVRelayEnv, build_training_scenarios


def evaluate_model(model, eval_scenarios, method_name):
    """Evaluate a trained RL model on 100 canonical scenarios."""
    results_per_scenario = []
    for scenario in eval_scenarios:
        obs = extract_features(scenario)
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        inference_ms = (time.perf_counter() - t0) * 1000

        pos = np.array([
            np.clip(action[0], 0, 100),
            np.clip(action[1], 0, 100),
            np.clip(action[2], 10, 40),
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
    print("=" * 60)
    print("SAC 500K TIMESTEP TRAINING")
    print("=" * 60)

    # Build training and eval scenario pools
    train_scenarios = build_training_scenarios()
    eval_scenarios = generate_scenarios(100)

    # Analytical baseline for comparison
    analytical_tp = []
    for s in eval_scenarios:
        pos = get_position_analytical(s)
        m = compute_channel_metrics(pos, s)
        analytical_tp.append(m['total_throughput'])
    analytical_mean = float(np.mean(analytical_tp))
    print(f"Analytical baseline: {analytical_mean:.1f} Mbps")

    # Create 8 parallel environments
    N_ENVS = 8
    TOTAL_TIMESTEPS = 500_000

    def make_env():
        return UAVRelayEnv(train_scenarios)

    env = DummyVecEnv([make_env for _ in range(N_ENVS)])

    # Train SAC
    print(f"\nTraining SAC with {TOTAL_TIMESTEPS:,} timesteps, {N_ENVS} parallel envs...")
    t0 = time.time()
    sac_model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=8,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 128]),
        device='cpu',
        seed=42,
    )
    sac_model.learn(total_timesteps=TOTAL_TIMESTEPS)
    train_time = time.time() - t0
    print(f"\nSAC training completed in {train_time:.1f}s ({train_time/60:.1f} min)")

    # Save trained model
    model_dir = '/home/it-services/ros2_ws/src/vla_6g_tvt/results'
    model_path = os.path.join(model_dir, 'sac_500k_model')
    sac_model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate on 100 canonical scenarios
    print(f"\nEvaluating SAC (500K) on 100 canonical scenarios...")
    sac_summary, sac_per_scenario = evaluate_model(sac_model, eval_scenarios, "sac_500k")

    # Assemble results
    results = {
        'method': 'sac_500k',
        'architecture': 'MlpPolicy [256, 128] (SAC)',
        'training_timesteps': TOTAL_TIMESTEPS,
        'training_time_s': train_time,
        'num_parallel_envs': N_ENVS,
        'hyperparameters': {
            'learning_rate': 3e-4,
            'buffer_size': 100_000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 8,
            'gradient_steps': 1,
            'seed': 42,
            'device': 'cpu',
        },
        'summary': sac_summary,
        'per_scenario': sac_per_scenario,
        'comparison': {
            'analytical_mean_throughput': analytical_mean,
            'sac_500k_vs_analytical_pct': float(
                (sac_summary['throughput_mbps']['mean'] - analytical_mean)
                / analytical_mean * 100
            ),
        },
    }

    # Print results
    print(f"\n{'=' * 60}")
    print("SAC 500K RESULTS")
    print(f"{'=' * 60}")
    print(f"  Throughput:  {sac_summary['throughput_mbps']['mean']:.1f} +/- "
          f"{sac_summary['throughput_mbps']['ci95']:.1f} Mbps")
    print(f"  Fairness:    {sac_summary['fairness_mean']:.3f}")
    print(f"  Coverage:    {sac_summary['coverage_mean']*100:.1f}%")
    print(f"  Latency:     {sac_summary['latency_ms']['mean']:.3f} ms "
          f"(p50={sac_summary['latency_ms']['p50']:.3f}, "
          f"p95={sac_summary['latency_ms']['p95']:.3f})")
    print(f"  vs Analytical: {results['comparison']['sac_500k_vs_analytical_pct']:+.1f}%")
    print(f"  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Cross-method comparison
    print(f"\n{'=' * 60}")
    print("CROSS-METHOD COMPARISON")
    print(f"{'=' * 60}")
    print(f"  SAC  (500K steps): {sac_summary['throughput_mbps']['mean']:.1f} Mbps")
    print(f"  SAC  (100K steps): ~71 Mbps (previous run)")
    print(f"  PPO  (100K steps): 68.6 Mbps (previous run)")
    print(f"  MLP  (8K samples): 168.9 Mbps (previous run)")
    print(f"  VLA  (2K samples): 134.0 Mbps (previous run)")
    print(f"  Analytical:        {analytical_mean:.1f} Mbps")

    # Save results
    path = save_results(results, 'sac_500k_baseline')
    print(f"\nDone! Results saved to {path}")


if __name__ == '__main__':
    main()
