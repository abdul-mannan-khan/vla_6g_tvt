#!/usr/bin/env python3
"""
M3: PPO (DRL) Baseline — Train & Evaluate

Env: Single-step episodes.
  Obs = 37-dim features (same as MLP).
  Action = Box([0,0,10], [100,100,40]) — UAV position.
  Reward = composite_score / 1000.
Training: PPO, MlpPolicy [256,128], 100k timesteps, CPU only.
Evaluate: Same 100 canonical scenarios, deterministic policy.
"""

import sys
import os
import json
import time
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (
    generate_scenarios, compute_channel_metrics, composite_score,
    extract_features, get_position_analytical,
    save_results, FEATURE_DIM, DATA_DIR,
)


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class UAVRelayEnv(gym.Env):
    """Single-step UAV relay positioning environment for PPO training.

    Each episode:
      1. Sample a random scenario from the training pool
      2. Agent outputs a 3D position (action)
      3. Reward = composite_score(metrics) / 1000
      4. Episode terminates immediately (single step)
    """
    metadata = {'render_modes': []}

    def __init__(self, scenarios):
        super().__init__()
        self.scenarios = scenarios
        self._rng = np.random.default_rng(42)
        self._current = None

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(FEATURE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 10.0], dtype=np.float32),
            high=np.array([100.0, 100.0, 40.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self._rng.integers(0, len(self.scenarios))
        self._current = self.scenarios[idx]
        obs = extract_features(self._current)
        return obs, {}

    def step(self, action):
        pos = np.array([
            np.clip(action[0], 0, 100),
            np.clip(action[1], 0, 100),
            np.clip(action[2], 10, 40),
        ])
        metrics = compute_channel_metrics(pos, self._current)
        reward = composite_score(metrics) / 1000.0
        return extract_features(self._current), reward, True, False, {
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage': metrics['coverage_rate'],
        }


# ---------------------------------------------------------------------------
# Training scenario pool (use training data to generate diverse scenarios)
# ---------------------------------------------------------------------------

def build_training_scenarios():
    """Build a large pool of training scenarios from the training data."""
    data_path = os.path.join(DATA_DIR, 'vla_training_data_20260130_202752.json')
    print(f"Loading training data from {data_path}...")
    with open(data_path) as f:
        raw = json.load(f)

    from eval_common import Scenario
    scenarios = []
    for s in raw['samples'][:8000]:
        scenarios.append(Scenario(
            id=s['sample_id'],
            num_users=s['num_users'],
            user_positions=[np.array(p) for p in s['user_positions']],
            user_requirements=[25.0] * s['num_users'],
            bs_position=np.array(s['bs_position']),
            initial_uav_position=np.array(s['uav_position']),
        ))
    print(f"Built {len(scenarios)} training scenarios for PPO")
    return scenarios


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Build training scenarios
    train_scenarios = build_training_scenarios()

    # Create env
    def make_env():
        return UAVRelayEnv(train_scenarios)

    env = DummyVecEnv([make_env])

    # Train PPO
    print("\nTraining PPO (100k timesteps, CPU)...")
    t0 = time.time()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 128]),
        device='cpu',
        seed=42,
    )
    model.learn(total_timesteps=100_000)
    train_time = time.time() - t0
    print(f"PPO training completed in {train_time:.1f}s")

    # Save model
    model_path = os.path.join(
        '/home/it-services/ros2_ws/src/vla_6g_tvt/results', 'ppo_model')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate on canonical 100 scenarios
    print("\nEvaluating PPO on 100 canonical scenarios...")
    eval_scenarios = generate_scenarios(100)

    results_per_scenario = []
    for scenario in eval_scenarios:
        obs = extract_features(scenario)

        t0_inf = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        inference_ms = (time.perf_counter() - t0_inf) * 1000

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

    # Aggregate
    tp_arr = np.array([r['throughput'] for r in results_per_scenario])
    fair_arr = np.array([r['fairness'] for r in results_per_scenario])
    cov_arr = np.array([r['coverage_rate'] for r in results_per_scenario])
    lat_arr = np.array([r['inference_ms'] for r in results_per_scenario])

    # Analytical baseline comparison
    analytical_tp = []
    for s in eval_scenarios:
        pos = get_position_analytical(s)
        m = compute_channel_metrics(pos, s)
        analytical_tp.append(m['total_throughput'])

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

    results = {
        'method': 'ppo',
        'architecture': 'MlpPolicy [256, 128]',
        'training_timesteps': 100000,
        'training_time_s': train_time,
        'summary': summary,
        'per_scenario': results_per_scenario,
        'comparison': {
            'analytical_mean_throughput': float(np.mean(analytical_tp)),
            'ppo_vs_analytical_pct': float(
                (summary['throughput_mbps']['mean'] - np.mean(analytical_tp))
                / np.mean(analytical_tp) * 100
            ),
        },
    }

    print(f"\n{'='*60}")
    print("PPO BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"  Throughput: {summary['throughput_mbps']['mean']:.1f} +/- "
          f"{summary['throughput_mbps']['ci95']:.1f} Mbps")
    print(f"  Fairness:   {summary['fairness_mean']:.3f}")
    print(f"  Coverage:   {summary['coverage_mean']*100:.1f}%")
    print(f"  Latency:    {summary['latency_ms']['mean']:.3f} ms")
    print(f"  vs Analytical: {results['comparison']['ppo_vs_analytical_pct']:+.1f}%")

    path = save_results(results, 'drl_baseline')
    print(f"\nDone! Results at {path}")


if __name__ == '__main__':
    main()
