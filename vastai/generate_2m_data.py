#!/usr/bin/env python3
"""
Generate 2M Training Data Points for VLA

Uses multiprocessing to parallelize DE optimization.
Designed to run on Vast.ai with multiple CPU cores.

Estimated time on 32-core machine: ~4-6 hours for 2M samples
"""

import os
import sys
import json
import random
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import differential_evolution

# Constants
FREQ_GHZ = 300
BANDWIDTH_GHZ = 10
BS_POWER_DBM = 30
UAV_POWER_DBM = 20
NOISE_FLOOR_DBM = -174 + 10 * np.log10(BANDWIDTH_GHZ * 1e9)
ABSORPTION_DB_PER_KM = 10
MAX_USERS = 7


@dataclass
class Scenario:
    id: int
    num_users: int
    user_positions: List[np.ndarray]
    user_requirements: List[float]
    bs_position: np.ndarray
    initial_uav_position: np.ndarray


def calc_snr(distance: float, power_dbm: float) -> float:
    """Calculate SNR for THz link."""
    if distance < 0.1:
        distance = 0.1
    fspl = 20 * np.log10(distance) + 20 * np.log10(FREQ_GHZ * 1e9) + 92.45
    absorption = ABSORPTION_DB_PER_KM * (distance / 1000)
    path_loss = fspl + absorption
    return power_dbm - path_loss - NOISE_FLOOR_DBM


def compute_metrics(uav_pos: np.ndarray, scenario: Scenario) -> dict:
    """Compute channel metrics for a UAV position."""
    d_bs_uav = np.linalg.norm(uav_pos - scenario.bs_position)
    snr_bs_uav = calc_snr(d_bs_uav, BS_POWER_DBM)

    user_rates = []
    users_covered = 0
    user_snrs = []

    for i, user_pos in enumerate(scenario.user_positions):
        d_uav_user = np.linalg.norm(uav_pos - user_pos)
        snr_uav_user = calc_snr(d_uav_user, UAV_POWER_DBM)
        effective_snr = min(snr_bs_uav, snr_uav_user)
        user_snrs.append(effective_snr)

        snr_linear = 10 ** (effective_snr / 10)
        rate = BANDWIDTH_GHZ * 1000 * np.log2(1 + max(snr_linear, 0.001))
        rate = min(rate, 10000)
        user_rates.append(rate)

        if rate >= scenario.user_requirements[i]:
            users_covered += 1

    total_throughput = sum(user_rates)

    if user_rates:
        s = sum(user_rates)
        s2 = sum(r ** 2 for r in user_rates)
        n = len(user_rates)
        fairness = (s ** 2) / (n * s2) if s2 > 0 else 0
    else:
        fairness = 0

    return {
        'total_throughput': total_throughput,
        'fairness': fairness,
        'coverage_rate': users_covered / len(user_rates),
        'user_rates': user_rates,
        'user_snrs': user_snrs,
    }


def composite_score(metrics: dict) -> float:
    """Compute composite score."""
    return (0.6 * metrics['total_throughput'] +
            0.3 * metrics['fairness'] * 1000 +
            0.1 * metrics['coverage_rate'] * 1000)


def generate_scenario(seed: int) -> Scenario:
    """Generate a random scenario."""
    rng = np.random.RandomState(seed)
    random.seed(seed)

    num_users = rng.choice([3, 4, 5, 6, 7])
    bs_position = np.array([0.0, 0.0, 30.0])

    # Random topology type
    topology = seed % 4

    user_positions = []
    if topology == 0:  # Clustered
        center = np.array([rng.uniform(40, 60), rng.uniform(40, 60)])
        for _ in range(num_users):
            pos = np.array([
                center[0] + rng.normal(0, 5),
                center[1] + rng.normal(0, 5),
                1.0
            ])
            pos[0] = np.clip(pos[0], 10, 90)
            pos[1] = np.clip(pos[1], 10, 90)
            user_positions.append(pos)
    elif topology == 1:  # Spread
        for _ in range(num_users):
            user_positions.append(np.array([
                rng.uniform(20, 80),
                rng.uniform(20, 80),
                1.0
            ]))
    elif topology == 2:  # Linear
        y_pos = rng.uniform(30, 70)
        for i in range(num_users):
            user_positions.append(np.array([
                20 + i * 60 / max(num_users - 1, 1),
                y_pos + rng.uniform(-5, 5),
                1.0
            ]))
    else:  # Circular
        center = np.array([50, 50])
        radius = rng.uniform(15, 30)
        for i in range(num_users):
            angle = 2 * np.pi * i / num_users
            user_positions.append(np.array([
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                1.0
            ]))

    user_requirements = [rng.uniform(10, 50) for _ in range(num_users)]
    initial_uav = np.array([25.0, 25.0, 20.0])

    return Scenario(
        id=seed,
        num_users=num_users,
        user_positions=user_positions,
        user_requirements=user_requirements,
        bs_position=bs_position,
        initial_uav_position=initial_uav,
    )


def solve_scenario(args: Tuple[int, int]) -> dict:
    """Solve a single scenario using DE optimizer."""
    seed, idx = args

    scenario = generate_scenario(seed)

    def objective(pos_array):
        uav_pos = np.array([
            np.clip(pos_array[0], 0, 100),
            np.clip(pos_array[1], 0, 100),
            np.clip(pos_array[2], 10, 40),
        ])
        metrics = compute_metrics(uav_pos, scenario)
        return -composite_score(metrics)

    t0 = time.time()
    result = differential_evolution(
        objective,
        bounds=[(0, 100), (0, 100), (10, 40)],
        maxiter=100,
        popsize=15,
        seed=seed,
        tol=0.01,
        polish=False,
    )
    solve_time_ms = (time.time() - t0) * 1000

    optimal_pos = np.array([
        np.clip(result.x[0], 0, 100),
        np.clip(result.x[1], 0, 100),
        np.clip(result.x[2], 10, 40),
    ])

    metrics = compute_metrics(optimal_pos, scenario)
    initial_metrics = compute_metrics(scenario.initial_uav_position, scenario)

    # Format instruction
    instruction = f"""You are a 6G UAV relay positioning expert. Given the current network state, determine the optimal 3D position for the UAV relay to maximize throughput and fairness.

Current State:
- Base station: ({scenario.bs_position[0]:.1f}, {scenario.bs_position[1]:.1f}, {scenario.bs_position[2]:.1f})
- UAV position: ({scenario.initial_uav_position[0]:.1f}, {scenario.initial_uav_position[1]:.1f}, {scenario.initial_uav_position[2]:.1f})
- Total throughput: {initial_metrics['total_throughput']:.1f} Mbps
- Fairness index: {initial_metrics['fairness']:.3f}

Ground Users ({scenario.num_users} total):"""

    for i, (pos, snr, rate) in enumerate(zip(
        scenario.user_positions,
        initial_metrics['user_snrs'],
        initial_metrics['user_rates']
    )):
        instruction += f"\n  User {i}: position=({pos[0]:.1f}, {pos[1]:.1f}), SNR={snr:.1f}dB, rate={rate:.1f}Mbps"

    instruction += "\n\nProvide the optimal UAV position and explain your reasoning."

    # Format output
    output = json.dumps({
        "x": round(optimal_pos[0], 1),
        "y": round(optimal_pos[1], 1),
        "z": round(optimal_pos[2], 1),
        "reasoning": f"Optimization-based positioning via differential evolution. Position ({optimal_pos[0]:.1f}, {optimal_pos[1]:.1f}, {optimal_pos[2]:.1f}). Score={-result.fun:.1f} (tp={metrics['total_throughput']:.1f}, fair={metrics['fairness']:.3f}, cov={metrics['coverage_rate']*100:.1f}%). Solved in {solve_time_ms:.0f}ms.",
        "score": round(-result.fun, 1)
    })

    return {
        "instruction": instruction,
        "input": "",
        "output": output
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate VLA training data')
    parser.add_argument('--num_samples', type=int, default=2000000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='vla_training_2m.jsonl', help='Output file')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--start_seed', type=int, default=0, help='Starting seed')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for progress reporting')
    args = parser.parse_args()

    num_workers = args.workers or cpu_count()
    print(f"=" * 60)
    print(f"VLA Training Data Generation")
    print(f"=" * 60)
    print(f"Samples to generate: {args.num_samples:,}")
    print(f"Output file: {args.output}")
    print(f"Parallel workers: {num_workers}")
    print(f"Starting seed: {args.start_seed}")

    # Estimate time
    samples_per_second = num_workers * 3  # ~3 samples/sec per worker
    estimated_hours = args.num_samples / samples_per_second / 3600
    print(f"Estimated time: {estimated_hours:.1f} hours")
    print(f"=" * 60)

    start_time = time.time()

    with open(args.output, 'w') as f:
        with Pool(num_workers) as pool:
            # Process in batches for progress reporting
            for batch_start in range(0, args.num_samples, args.batch_size):
                batch_end = min(batch_start + args.batch_size, args.num_samples)
                batch_args = [(args.start_seed + i, i) for i in range(batch_start, batch_end)]

                results = pool.map(solve_scenario, batch_args)

                for result in results:
                    f.write(json.dumps(result) + '\n')

                elapsed = time.time() - start_time
                rate = (batch_end) / elapsed
                eta = (args.num_samples - batch_end) / rate if rate > 0 else 0

                print(f"Progress: {batch_end:,}/{args.num_samples:,} ({100*batch_end/args.num_samples:.1f}%) "
                      f"| Rate: {rate:.1f} samples/s | ETA: {eta/3600:.1f}h")

    total_time = time.time() - start_time
    print(f"=" * 60)
    print(f"COMPLETE!")
    print(f"Generated {args.num_samples:,} samples in {total_time/3600:.2f} hours")
    print(f"Output: {args.output}")
    print(f"=" * 60)


if __name__ == '__main__':
    main()
