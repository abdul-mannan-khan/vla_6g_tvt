#!/usr/bin/env python3
"""
Run VLA (LM-Relay) and TD3 on the 200 scenarios from exp1, then merge results
into the existing JSON.
"""

import sys
import os
import gc
import json
import time
import numpy as np
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# Add scripts dir to path for imports from evaluate_access_v2
sys.path.insert(0, SCRIPTS_DIR)
from evaluate_access_v2 import (
    generate_scenarios, VLALoader, DRLLoader,
    _compute_metrics_fspl, ci95, bootstrap_ci,
    MODEL_DIR, SAC_MODEL_PATH,
)

# Find existing exp1 results
import glob
exp1_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'access_v2_exp1_*.json')))
if not exp1_files:
    print("ERROR: No exp1 results found. Run evaluate_access_v2.py --exp 1 first.")
    sys.exit(1)

exp1_path = exp1_files[-1]
print(f"Loading existing results: {exp1_path}")
with open(exp1_path) as f:
    results = json.load(f)

# Generate same scenarios
scenarios = generate_scenarios(200, seed=2026)
print(f"Generated {len(scenarios)} scenarios (seed=2026)")

# ---- 1. VLA (LM-Relay) ----
print("\n--- LM-Relay (VLA TinyLlama + LoRA) ---")
try:
    vla = VLALoader(MODEL_DIR)
    vla.load(merge=True)

    vla_results = []
    parse_ok = 0
    parse_fail = 0
    latencies = []

    for idx, s in enumerate(scenarios):
        pos, parsed, latency = vla.predict(s)
        m = _compute_metrics_fspl(pos, s)
        m['parsed'] = parsed
        m['latency_s'] = latency
        vla_results.append(m)
        latencies.append(latency * 1000)  # ms
        if parsed:
            parse_ok += 1
        else:
            parse_fail += 1
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/200] tp={m['total_throughput']:.1f} "
                  f"lat={latency*1000:.0f}ms parsed={'Y' if parsed else 'N'}")

    tp_mean, tp_ci = ci95([r['total_throughput'] for r in vla_results])
    print(f"  Throughput: {tp_mean:.1f} +/- {tp_ci:.1f} Mbps")
    print(f"  Parse rate: {parse_ok}/{parse_ok+parse_fail} "
          f"({100*parse_ok/(parse_ok+parse_fail):.1f}%)")
    print(f"  Mean latency: {np.mean(latencies):.0f}ms")

    # Update results
    results['per_scenario']['LM-Relay'] = [
        {'scenario_id': i, 'throughput': r['total_throughput'],
         'fairness': r['fairness'], 'coverage_rate': r['coverage_rate']}
        for i, r in enumerate(vla_results)
    ]

    rng_bs = np.random.default_rng(2026)
    tp_vals = [r['total_throughput'] for r in vla_results]
    fair_vals = [r['fairness'] for r in vla_results]
    cov_vals = [r['coverage_rate'] for r in vla_results]
    _, tp_lo, tp_hi = bootstrap_ci(tp_vals, rng=rng_bs)

    results['summary']['LM-Relay'] = {
        'throughput_mean': np.mean(tp_vals),
        'throughput_ci': 1.96 * np.std(tp_vals, ddof=1) / np.sqrt(len(tp_vals)),
        'throughput_std': np.std(tp_vals, ddof=1),
        'throughput_bootstrap_lo': tp_lo,
        'throughput_bootstrap_hi': tp_hi,
        'fairness_mean': np.mean(fair_vals),
        'fairness_ci': 1.96 * np.std(fair_vals, ddof=1) / np.sqrt(len(fair_vals)),
        'coverage_mean': np.mean(cov_vals),
        'coverage_ci': 1.96 * np.std(cov_vals, ddof=1) / np.sqrt(len(cov_vals)),
        'parse_success_rate': parse_ok / (parse_ok + parse_fail),
        'mean_latency_ms': np.mean(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
    }

    vla.unload()

except Exception as e:
    print(f"  VLA failed: {e}")
    import traceback; traceback.print_exc()

# ---- 2. TD3 (train from scratch using stable-baselines3) ----
print("\n--- TD3 (training from scratch, 50K steps) ---")
try:
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    import gymnasium as gym
    from gymnasium import spaces

    # Simple relay environment
    class RelayEnv(gym.Env):
        def __init__(self, scenarios_list):
            super().__init__()
            self.scenarios = scenarios_list
            self.idx = 0
            # observation: [bs_x, bs_y, bs_z, uav_x, uav_y, uav_z, num_users,
            #               mean_user_x, mean_user_y]
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
            # action: [x, y, z] normalized to [-1, 1]
            self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        def _get_obs(self):
            s = self.scenarios[self.idx]
            user_x = np.mean([u[0] for u in s.user_positions])
            user_y = np.mean([u[1] for u in s.user_positions])
            return np.array([
                s.bs_position[0], s.bs_position[1], s.bs_position[2],
                s.initial_uav_position[0], s.initial_uav_position[1], s.initial_uav_position[2],
                float(s.num_users), user_x, user_y
            ], dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.idx = np.random.randint(len(self.scenarios))
            return self._get_obs(), {}

        def step(self, action):
            pos = np.array([
                np.clip(action[0] * 50 + 50, 0, 100),
                np.clip(action[1] * 50 + 50, 0, 100),
                np.clip(action[2] * 15 + 25, 10, 40),
            ])
            m = _compute_metrics_fspl(pos, self.scenarios[self.idx])
            reward = (m['total_throughput'] / 100.0 + m['fairness'] + m['coverage_rate']) / 3.0
            self.idx = np.random.randint(len(self.scenarios))
            return self._get_obs(), reward, True, False, {}

    env = RelayEnv(scenarios)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    td3_model = TD3(
        "MlpPolicy", env, action_noise=action_noise,
        learning_rate=3e-4, buffer_size=50000,
        learning_starts=1000, batch_size=256,
        tau=0.005, gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 128]),
        verbose=0, seed=2026)

    print("  Training TD3 (50K steps)...")
    t0 = time.time()
    td3_model.learn(total_timesteps=50000)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")

    # Save the model
    td3_path = os.path.join(RESULTS_DIR, 'td3_50k_model')
    td3_model.save(td3_path)
    print(f"  Saved to {td3_path}")

    # Evaluate
    td3_results = []
    for s in scenarios:
        from evaluate_access_v2 import extract_features
        feat = extract_features(s)
        action, _ = td3_model.predict(feat, deterministic=True)
        pos = np.array([
            np.clip(action[0], 0, 100),
            np.clip(action[1], 0, 100),
            np.clip(action[2], 10, 40),
        ])
        m = _compute_metrics_fspl(pos, s)
        td3_results.append(m)

    tp_mean, tp_ci = ci95([r['total_throughput'] for r in td3_results])
    print(f"  Throughput: {tp_mean:.1f} +/- {tp_ci:.1f} Mbps")

    results['per_scenario']['TD3'] = [
        {'scenario_id': i, 'throughput': r['total_throughput'],
         'fairness': r['fairness'], 'coverage_rate': r['coverage_rate']}
        for i, r in enumerate(td3_results)
    ]

    rng_bs2 = np.random.default_rng(2026)
    tp_vals = [r['total_throughput'] for r in td3_results]
    fair_vals = [r['fairness'] for r in td3_results]
    cov_vals = [r['coverage_rate'] for r in td3_results]
    _, tp_lo, tp_hi = bootstrap_ci(tp_vals, rng=rng_bs2)

    results['summary']['TD3'] = {
        'throughput_mean': np.mean(tp_vals),
        'throughput_ci': 1.96 * np.std(tp_vals, ddof=1) / np.sqrt(len(tp_vals)),
        'throughput_std': np.std(tp_vals, ddof=1),
        'throughput_bootstrap_lo': tp_lo,
        'throughput_bootstrap_hi': tp_hi,
        'fairness_mean': np.mean(fair_vals),
        'fairness_ci': 1.96 * np.std(fair_vals, ddof=1) / np.sqrt(len(fair_vals)),
        'coverage_mean': np.mean(cov_vals),
        'coverage_ci': 1.96 * np.std(cov_vals, ddof=1) / np.sqrt(len(cov_vals)),
    }

    del td3_model

except Exception as e:
    print(f"  TD3 failed: {e}")
    import traceback; traceback.print_exc()

# ---- Recompute statistics ----
print("\n--- Recomputing statistical tests ---")
lm_tp = [r['throughput'] for r in results['per_scenario']['LM-Relay']]
stat_tests = {}
from scipy import stats as sp_stats
for method in results['methods']:
    if method == 'LM-Relay':
        continue
    m_tp = [r['throughput'] for r in results['per_scenario'][method]]
    t_stat, p_val = sp_stats.ttest_rel(lm_tp, m_tp)
    d = np.mean(np.array(lm_tp) - np.array(m_tp)) / np.std(np.array(lm_tp) - np.array(m_tp), ddof=1)
    n_tests = len(results['methods']) - 1
    p_corr = min(p_val * n_tests, 1.0)
    sig = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'
    stat_tests[method] = {
        't_stat': float(t_stat), 'p_value': float(p_val),
        'p_corrected': float(p_corr), 'cohens_d': float(d), 'sig': sig
    }
    print(f"  vs {method:18s}: t={t_stat:7.2f}, p_corr={p_corr:.4f}, d={d:.3f} {sig}")

results['summary']['statistical_tests'] = stat_tests

# ---- Recompute per-topology ----
print("\n--- Recomputing per-topology breakdown ---")
topology_names = ['clustered', 'spread', 'linear', 'circular']
per_topo = {}
for topo_idx, topo_name in enumerate(topology_names):
    start = topo_idx * 50
    end = start + 50
    per_topo[topo_name] = {}
    for method in results['methods']:
        tp_slice = [results['per_scenario'][method][i]['throughput'] for i in range(start, end)]
        fair_slice = [results['per_scenario'][method][i]['fairness'] for i in range(start, end)]
        cov_slice = [results['per_scenario'][method][i]['coverage_rate'] for i in range(start, end)]
        per_topo[topo_name][method] = {
            'throughput_mean': float(np.mean(tp_slice)),
            'fairness_mean': float(np.mean(fair_slice)),
            'coverage_mean': float(np.mean(cov_slice)),
        }
    print(f"  {topo_name}: LM-Relay tp={per_topo[topo_name]['LM-Relay']['throughput_mean']:.1f}, "
          f"TD3 tp={per_topo[topo_name]['TD3']['throughput_mean']:.1f}")

results['summary']['per_topology'] = per_topo

# Save updated results
out_path = os.path.join(RESULTS_DIR, f'access_v2_exp1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nUpdated results saved to {out_path}")

# Print final summary
print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY")
print("=" * 70)
for method in results['methods']:
    s = results['summary'][method]
    print(f"  {method:18s}: tp={s['throughput_mean']:6.1f} +/- {s['throughput_ci']:.1f}  "
          f"fair={s['fairness_mean']:.3f}  cov={100*s['coverage_mean']:.1f}%")
