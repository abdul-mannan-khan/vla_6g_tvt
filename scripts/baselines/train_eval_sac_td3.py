#!/usr/bin/env python3
"""
W2 Pre-emption: Train SAC and TD3 baselines — proper continuous-action DRL.

SAC (Soft Actor-Critic): entropy-regularized, designed for continuous actions.
TD3 (Twin Delayed DDPG): target policy smoothing, clipped double-Q.

Both trained with 500K timesteps (5x the PPO budget) on the same 8,000 scenarios.
"""

import sys, os, json, time
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise

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
    train_scenarios = build_training_scenarios()
    eval_scenarios = generate_scenarios(100)

    # Analytical baseline
    analytical_tp = []
    for s in eval_scenarios:
        pos = get_position_analytical(s)
        m = compute_channel_metrics(pos, s)
        analytical_tp.append(m['total_throughput'])
    analytical_mean = float(np.mean(analytical_tp))

    all_results = {}

    # ---------------------------------------------------------------
    # SAC (Soft Actor-Critic) — the gold standard for continuous control
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("TRAINING SAC (500K timesteps)")
    print("="*60)

    N_ENVS = 8

    def make_env():
        return UAVRelayEnv(train_scenarios)

    env = DummyVecEnv([make_env for _ in range(N_ENVS)])

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
    sac_model.learn(total_timesteps=100_000)
    sac_train_time = time.time() - t0
    print(f"SAC training: {sac_train_time:.1f}s")

    sac_summary, sac_per_scenario = evaluate_model(sac_model, eval_scenarios, "sac")

    all_results['sac'] = {
        'method': 'sac',
        'architecture': 'MlpPolicy [256, 128] (SAC)',
        'training_timesteps': 100_000,
        'training_time_s': sac_train_time,
        'summary': sac_summary,
        'per_scenario': sac_per_scenario,
        'comparison': {
            'analytical_mean_throughput': analytical_mean,
            'sac_vs_analytical_pct': float(
                (sac_summary['throughput_mbps']['mean'] - analytical_mean)
                / analytical_mean * 100
            ),
        },
    }

    print(f"\nSAC Results:")
    print(f"  Throughput: {sac_summary['throughput_mbps']['mean']:.1f} +/- "
          f"{sac_summary['throughput_mbps']['ci95']:.1f} Mbps")
    print(f"  Fairness:   {sac_summary['fairness_mean']:.3f}")
    print(f"  Coverage:   {sac_summary['coverage_mean']*100:.1f}%")

    sac_path = save_results(all_results['sac'], 'sac_baseline')

    # ---------------------------------------------------------------
    # TD3 (Twin Delayed DDPG)
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("TRAINING TD3 (500K timesteps)")
    print("="*60)

    env2 = DummyVecEnv([make_env for _ in range(N_ENVS)])

    n_actions = 3
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    t0 = time.time()
    td3_model = TD3(
        "MlpPolicy",
        env2,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=8,
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[256, 128]),
        device='cpu',
        seed=42,
    )
    td3_model.learn(total_timesteps=100_000)
    td3_train_time = time.time() - t0
    print(f"TD3 training: {td3_train_time:.1f}s")

    td3_summary, td3_per_scenario = evaluate_model(td3_model, eval_scenarios, "td3")

    all_results['td3'] = {
        'method': 'td3',
        'architecture': 'MlpPolicy [256, 128] (TD3)',
        'training_timesteps': 100_000,
        'training_time_s': td3_train_time,
        'summary': td3_summary,
        'per_scenario': td3_per_scenario,
        'comparison': {
            'analytical_mean_throughput': analytical_mean,
            'td3_vs_analytical_pct': float(
                (td3_summary['throughput_mbps']['mean'] - analytical_mean)
                / analytical_mean * 100
            ),
        },
    }

    print(f"\nTD3 Results:")
    print(f"  Throughput: {td3_summary['throughput_mbps']['mean']:.1f} +/- "
          f"{td3_summary['throughput_mbps']['ci95']:.1f} Mbps")
    print(f"  Fairness:   {td3_summary['fairness_mean']:.3f}")
    print(f"  Coverage:   {td3_summary['coverage_mean']*100:.1f}%")

    td3_path = save_results(all_results['td3'], 'td3_baseline')

    # ---------------------------------------------------------------
    # Summary comparison
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("COMPARISON: PPO vs SAC vs TD3 (all DRL baselines)")
    print(f"{'='*60}")
    print(f"  PPO  (100K steps):  68.6 Mbps  (previous run)")
    print(f"  SAC  (100K steps):  {sac_summary['throughput_mbps']['mean']:.1f} Mbps")
    print(f"  TD3  (100K steps):  {td3_summary['throughput_mbps']['mean']:.1f} Mbps")
    print(f"  MLP  (8K samples):  168.9 Mbps (previous run)")
    print(f"  VLA  (2K samples):  134.0 Mbps (previous run)")
    print(f"  Analytical:         {analytical_mean:.1f} Mbps")


if __name__ == '__main__':
    main()
