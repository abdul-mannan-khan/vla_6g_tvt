#!/usr/bin/env python3
"""
Channel Optimizer for VLA-6G Training Ground Truth

Uses scipy.optimize.differential_evolution to find globally optimal
UAV relay positions, providing optimization-based ground truth that
exceeds the analytical heuristic.

Multi-objective: score = w1*throughput + w2*fairness + w3*coverage
"""

import numpy as np
from scipy.optimize import differential_evolution
from typing import List, Dict, Tuple
import time


class ChannelModel:
    """Reusable 6G THz channel model for SNR, throughput, and fairness."""

    def __init__(self, frequency_ghz: float = 300.0, bandwidth_ghz: float = 10.0,
                 bs_power_dbm: float = 30.0, uav_power_dbm: float = 20.0):
        self.frequency_ghz = frequency_ghz
        self.bandwidth_ghz = bandwidth_ghz
        self.bs_power_dbm = bs_power_dbm
        self.uav_power_dbm = uav_power_dbm

    def calc_snr(self, distance: float, power_dbm: float) -> float:
        """Calculate SNR in dB for a given link distance and transmit power."""
        d_km = max(distance / 1000.0, 0.001)
        fspl = 20 * np.log10(d_km) + 20 * np.log10(self.frequency_ghz) + 92.45
        absorption = 10.0 * d_km
        path_loss = fspl + absorption
        noise = -174 + 10 * np.log10(self.bandwidth_ghz * 1e9) + 10
        return power_dbm - path_loss - noise

    def compute_metrics(self, uav_pos: np.ndarray, bs_position: np.ndarray,
                        user_positions: List[np.ndarray],
                        user_requirements: List[float] = None) -> Dict:
        """Compute full channel metrics for a UAV position.

        Returns dict with: total_throughput, user_rates, fairness, coverage_rate,
        average_rate, min_rate, user_snrs
        """
        d_bs_uav = np.linalg.norm(uav_pos - bs_position)
        snr_bs_uav = self.calc_snr(d_bs_uav, self.bs_power_dbm)

        user_rates = []
        user_snrs = []
        users_covered = 0

        for i, user_pos in enumerate(user_positions):
            d_uav_user = np.linalg.norm(uav_pos - user_pos)
            snr_uav_user = self.calc_snr(d_uav_user, self.uav_power_dbm)
            effective_snr = min(snr_bs_uav, snr_uav_user)

            snr_linear = 10 ** (effective_snr / 10)
            rate = self.bandwidth_ghz * 1000 * np.log2(1 + max(snr_linear, 0.001))
            rate = min(rate, 10000)

            user_rates.append(rate)
            user_snrs.append(snr_uav_user)

            req = user_requirements[i] if user_requirements else 25.0
            if rate >= req:
                users_covered += 1

        total_throughput = sum(user_rates)
        n = len(user_rates)

        # Jain's fairness
        if user_rates:
            s = sum(user_rates)
            s2 = sum(r ** 2 for r in user_rates)
            fairness = (s ** 2) / (n * s2) if s2 > 0 else 0
        else:
            fairness = 0

        coverage_rate = users_covered / n if n > 0 else 0

        return {
            'total_throughput': total_throughput,
            'user_rates': user_rates,
            'user_snrs': user_snrs,
            'average_rate': np.mean(user_rates) if user_rates else 0,
            'min_rate': min(user_rates) if user_rates else 0,
            'fairness': fairness,
            'coverage_rate': coverage_rate,
        }


def optimize_position(bs_position: np.ndarray,
                      user_positions: List[np.ndarray],
                      user_requirements: List[float] = None,
                      weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
                      bounds: Tuple = ((0, 100), (0, 100), (10, 40))
                      ) -> Dict:
    """Find globally optimal UAV relay position using differential evolution.

    Args:
        bs_position: Base station [x, y, z]
        user_positions: List of user positions [x, y, z]
        user_requirements: Per-user QoS requirements (Mbps). Defaults to 25 each.
        weights: (w_throughput, w_fairness, w_coverage) for multi-objective score
        bounds: Search bounds for (x, y, z)

    Returns:
        Dict with: position, score, metrics, solve_time_ms, reasoning
    """
    if user_requirements is None:
        user_requirements = [25.0] * len(user_positions)

    channel = ChannelModel()
    w_tp, w_fair, w_cov = weights

    def objective(params):
        uav_pos = np.array(params)
        m = channel.compute_metrics(uav_pos, bs_position, user_positions, user_requirements)
        # Maximize: throughput (scaled) + fairness (scaled) + coverage (scaled)
        score = w_tp * m['total_throughput'] + w_fair * m['fairness'] * 1000 + w_cov * m['coverage_rate'] * 1000
        return -score  # minimize negative

    start = time.time()
    result = differential_evolution(
        objective,
        bounds=list(bounds),
        seed=42,
        maxiter=200,
        tol=1e-6,
        polish=True,
    )
    solve_time_ms = (time.time() - start) * 1000

    optimal_pos = np.array(result.x)
    metrics = channel.compute_metrics(optimal_pos, bs_position, user_positions, user_requirements)
    score = -result.fun

    reasoning = (
        f"Optimization-based positioning via differential evolution. "
        f"Position ({optimal_pos[0]:.1f}, {optimal_pos[1]:.1f}, {optimal_pos[2]:.1f}). "
        f"Score={score:.1f} (tp={metrics['total_throughput']:.1f}, "
        f"fair={metrics['fairness']:.3f}, cov={metrics['coverage_rate']:.1%}). "
        f"Solved in {solve_time_ms:.0f}ms."
    )

    return {
        'position': optimal_pos,
        'score': score,
        'metrics': metrics,
        'solve_time_ms': solve_time_ms,
        'reasoning': reasoning,
    }
