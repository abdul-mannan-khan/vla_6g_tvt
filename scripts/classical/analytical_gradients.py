#!/usr/bin/env python3
"""
Analytical Gradients for UAV Relay Positioning.

Provides closed-form gradients of the channel model for:
1. Physics-informed neural network training
2. Warm-starting RL policies
3. Lyapunov stability analysis

The key insight is that Shannon capacity C = BW * log2(1 + SNR) has
analytical derivatives w.r.t. position through the path loss model.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_common import (
    Scenario, compute_channel_metrics, calc_snr,
    FREQUENCY_GHZ, BANDWIDTH_GHZ, BS_POWER_DBM, UAV_POWER_DBM
)


def compute_snr_gradient(p: np.ndarray, anchor: np.ndarray,
                         power_dbm: float) -> Tuple[float, np.ndarray]:
    """
    Compute SNR and its gradient w.r.t. UAV position p.

    The THz channel model:
        SNR_dB = P_dBm - FSPL - absorption - noise
        FSPL = 20*log10(d_km) + 20*log10(f_GHz) + 92.45
        absorption = 10 * d_km

    Returns:
        snr_db: SNR in dB
        grad_snr_db: Gradient of SNR_dB w.r.t. position p
    """
    d = np.linalg.norm(p - anchor)
    d = max(d, 0.001)  # Avoid division by zero
    d_km = d / 1000.0

    # Compute SNR
    fspl = 20 * np.log10(d_km) + 20 * np.log10(FREQUENCY_GHZ) + 92.45
    absorption = 10.0 * d_km
    path_loss = fspl + absorption
    noise = -174 + 10 * np.log10(BANDWIDTH_GHZ * 1e9) + 10
    snr_db = power_dbm - path_loss - noise

    # Gradient of SNR_dB w.r.t. distance d
    # d(FSPL)/d(d) = 20 / (d * ln(10))
    # d(absorption)/d(d) = 10/1000 = 0.01
    d_fspl_d_d = 20 / (d * np.log(10))
    d_absorption_d_d = 10 / 1000
    d_snr_db_d_d = -(d_fspl_d_d + d_absorption_d_d)

    # d(d)/d(p) = (p - anchor) / d
    d_d_dp = (p - anchor) / d

    # Chain rule
    grad_snr_db = d_snr_db_d_d * d_d_dp

    return snr_db, grad_snr_db


def compute_rate_gradient(snr_db: float, grad_snr_db: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute rate and its gradient from SNR.

    Rate = BW * log2(1 + SNR_linear)
         = BW * log2(1 + 10^(SNR_dB/10))

    Returns:
        rate: Data rate in Mbps
        grad_rate: Gradient of rate w.r.t. position
    """
    snr_linear = 10 ** (snr_db / 10)
    rate = BANDWIDTH_GHZ * 1000 * np.log2(1 + max(snr_linear, 0.001))
    rate = min(rate, 10000)  # Cap at 10 Gbps

    # d(rate)/d(SNR_linear) = BW * 1000 / ((1 + SNR_linear) * ln(2))
    d_rate_d_snr_linear = BANDWIDTH_GHZ * 1000 / ((1 + snr_linear) * np.log(2))

    # d(SNR_linear)/d(SNR_dB) = SNR_linear * ln(10) / 10
    d_snr_linear_d_snr_db = snr_linear * np.log(10) / 10

    # Chain rule
    grad_rate = d_rate_d_snr_linear * d_snr_linear_d_snr_db * grad_snr_db

    return rate, grad_rate


def compute_throughput_gradient(p: np.ndarray, scenario: Scenario) -> Tuple[float, np.ndarray]:
    """
    Compute total throughput and its analytical gradient.

    For relay channel, effective SNR = min(SNR_BS_UAV, SNR_UAV_user)

    Returns:
        throughput: Total throughput in Mbps
        grad: Gradient of throughput w.r.t. position [3,]
    """
    # BS-UAV link
    snr_bs_uav, grad_snr_bs_uav = compute_snr_gradient(p, scenario.bs_position, BS_POWER_DBM)

    total_rate = 0.0
    total_grad = np.zeros(3)

    for user_pos in scenario.user_positions:
        # UAV-user link
        snr_uav_user, grad_snr_uav_user = compute_snr_gradient(p, user_pos, UAV_POWER_DBM)

        # Effective SNR is the bottleneck
        if snr_uav_user < snr_bs_uav:
            # UAV-user link is bottleneck
            rate, grad_rate = compute_rate_gradient(snr_uav_user, grad_snr_uav_user)
        else:
            # BS-UAV link is bottleneck
            rate, grad_rate = compute_rate_gradient(snr_bs_uav, grad_snr_bs_uav)

        total_rate += rate
        total_grad += grad_rate

    return total_rate, total_grad


def compute_fairness_gradient(p: np.ndarray, scenario: Scenario) -> Tuple[float, np.ndarray]:
    """
    Compute Jain's fairness index and its gradient.

    Fairness = (sum_i r_i)^2 / (n * sum_i r_i^2)

    Returns:
        fairness: Jain's fairness index [0,1]
        grad: Gradient of fairness w.r.t. position [3,]
    """
    # Compute all rates and their gradients
    rates = []
    grads = []

    snr_bs_uav, grad_snr_bs_uav = compute_snr_gradient(p, scenario.bs_position, BS_POWER_DBM)

    for user_pos in scenario.user_positions:
        snr_uav_user, grad_snr_uav_user = compute_snr_gradient(p, user_pos, UAV_POWER_DBM)

        if snr_uav_user < snr_bs_uav:
            rate, grad_rate = compute_rate_gradient(snr_uav_user, grad_snr_uav_user)
        else:
            rate, grad_rate = compute_rate_gradient(snr_bs_uav, grad_snr_bs_uav)

        rates.append(rate)
        grads.append(grad_rate)

    rates = np.array(rates)
    grads = np.array(grads)  # [n_users, 3]
    n = len(rates)

    S = np.sum(rates)
    S2 = np.sum(rates ** 2)

    if S2 < 1e-8:
        return 0.0, np.zeros(3)

    fairness = S ** 2 / (n * S2)

    # Gradient using quotient rule
    # d(fairness)/dp = d(S^2/(n*S2))/dp
    # = (2S * dS/dp * n * S2 - S^2 * n * dS2/dp) / (n*S2)^2
    # = (2S * sum_i dr_i/dp * S2 - S^2 * 2 * sum_i r_i * dr_i/dp) / (n * S2^2)

    dS_dp = np.sum(grads, axis=0)  # sum of gradients
    dS2_dp = 2 * np.sum(rates[:, None] * grads, axis=0)  # sum of 2*r_i * grad_r_i

    grad_fairness = (2 * S * dS_dp * S2 - S ** 2 * dS2_dp) / (n * S2 ** 2)

    return fairness, grad_fairness


def compute_composite_gradient(p: np.ndarray, scenario: Scenario,
                               weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)
                               ) -> Tuple[float, np.ndarray]:
    """
    Compute composite objective and its gradient.

    Composite = w1*throughput + w2*fairness*1000 + w3*coverage*1000

    Note: Coverage is not differentiable (step function), so we use
    a soft approximation or ignore its gradient.

    Returns:
        score: Composite score
        grad: Gradient of composite score w.r.t. position
    """
    w1, w2, w3 = weights

    throughput, grad_throughput = compute_throughput_gradient(p, scenario)
    fairness, grad_fairness = compute_fairness_gradient(p, scenario)

    # Coverage is non-differentiable, approximate as 0
    metrics = compute_channel_metrics(p, scenario)
    coverage = metrics['coverage_rate']

    score = w1 * throughput + w2 * fairness * 1000 + w3 * coverage * 1000

    # Gradient (ignoring coverage)
    grad = w1 * grad_throughput + w2 * 1000 * grad_fairness

    return score, grad


def compute_hessian_approx(p: np.ndarray, scenario: Scenario,
                           eps: float = 0.1) -> np.ndarray:
    """
    Compute approximate Hessian using finite differences.

    Used for second-order optimization and uncertainty estimation.

    Returns:
        H: Approximate Hessian matrix [3, 3]
    """
    H = np.zeros((3, 3))
    _, grad0 = compute_throughput_gradient(p, scenario)

    for i in range(3):
        p_plus = p.copy()
        p_plus[i] += eps

        # Clip to bounds
        if i < 2:
            p_plus[i] = np.clip(p_plus[i], 0, 100)
        else:
            p_plus[i] = np.clip(p_plus[i], 10, 40)

        _, grad_plus = compute_throughput_gradient(p_plus, scenario)
        H[:, i] = (grad_plus - grad0) / eps

    # Symmetrize
    H = (H + H.T) / 2

    return H


class GradientOracle:
    """
    Oracle providing gradients and Hessians for a scenario.

    Used by physics-informed RL to incorporate analytical gradients.
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def throughput_and_grad(self, p: np.ndarray) -> Tuple[float, np.ndarray]:
        """Get throughput and gradient at position p."""
        return compute_throughput_gradient(p, self.scenario)

    def fairness_and_grad(self, p: np.ndarray) -> Tuple[float, np.ndarray]:
        """Get fairness and gradient at position p."""
        return compute_fairness_gradient(p, self.scenario)

    def composite_and_grad(self, p: np.ndarray) -> Tuple[float, np.ndarray]:
        """Get composite score and gradient at position p."""
        return compute_composite_gradient(p, self.scenario)

    def hessian(self, p: np.ndarray) -> np.ndarray:
        """Get approximate Hessian at position p."""
        return compute_hessian_approx(p, self.scenario)

    def optimal_direction(self, p: np.ndarray) -> np.ndarray:
        """Get normalized gradient direction (for policy guidance)."""
        _, grad = compute_throughput_gradient(p, self.scenario)
        norm = np.linalg.norm(grad)
        if norm > 1e-6:
            return grad / norm
        return np.zeros(3)


if __name__ == "__main__":
    """Test analytical gradients against numerical."""
    from eval_common import generate_scenarios

    scenarios = generate_scenarios(num_scenarios=5)

    print("Testing Analytical Gradients")
    print("=" * 60)

    for scenario in scenarios[:3]:
        p = scenario.initial_uav_position.copy()

        # Analytical gradient
        throughput, grad_analytical = compute_throughput_gradient(p, scenario)

        # Numerical gradient (finite differences)
        eps = 0.001
        grad_numerical = np.zeros(3)
        for i in range(3):
            p_plus = p.copy()
            p_minus = p.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            t_plus, _ = compute_throughput_gradient(p_plus, scenario)
            t_minus, _ = compute_throughput_gradient(p_minus, scenario)
            grad_numerical[i] = (t_plus - t_minus) / (2 * eps)

        # Compare
        error = np.linalg.norm(grad_analytical - grad_numerical)
        rel_error = error / (np.linalg.norm(grad_numerical) + 1e-8)

        print(f"Scenario {scenario.id}:")
        print(f"  Throughput: {throughput:.2f} Mbps")
        print(f"  Analytical grad: [{grad_analytical[0]:.4f}, {grad_analytical[1]:.4f}, {grad_analytical[2]:.4f}]")
        print(f"  Numerical grad:  [{grad_numerical[0]:.4f}, {grad_numerical[1]:.4f}, {grad_numerical[2]:.4f}]")
        print(f"  Relative error: {rel_error:.2e}")
        print()
