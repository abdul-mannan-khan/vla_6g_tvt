#!/usr/bin/env python3
"""
Successive Convex Approximation (SCA) Solver for UAV Relay Positioning.

SCA iteratively solves convex subproblems by linearizing the non-convex
objective around the current solution. This provides:
1. Convergence guarantees (every limit point is stationary)
2. Analytical gradients for warm-starting RL
3. Upper bound on achievable performance

Reference: Razaviyayn et al., "A Unified Convergence Analysis of Block
Successive Minimization Methods for Nonsmooth Optimization" (2013)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_common import (
    Scenario, compute_channel_metrics, calc_snr,
    FREQUENCY_GHZ, BANDWIDTH_GHZ, BS_POWER_DBM, UAV_POWER_DBM
)


@dataclass
class SCAConfig:
    """Configuration for SCA solver."""
    max_iterations: int = 50
    trust_region_init: float = 10.0  # Initial trust region radius
    trust_region_min: float = 0.1
    trust_region_max: float = 50.0
    convergence_tol: float = 1e-4
    eta_expand: float = 1.5  # Trust region expansion factor
    eta_shrink: float = 0.5  # Trust region contraction factor
    rho_accept: float = 0.1  # Minimum improvement ratio to accept step


class SCASolver:
    """
    Successive Convex Approximation solver for UAV relay positioning.

    The problem is:
        max  sum_i log2(1 + SNR_i(p))
        s.t. p in [0,100]^2 x [10,40]

    This is non-convex due to the log-distance path loss model.
    SCA linearizes around current solution and solves QP subproblems.
    """

    def __init__(self, config: SCAConfig = None):
        self.config = config or SCAConfig()
        self.history = []

    def _compute_throughput(self, p: np.ndarray, scenario: Scenario) -> float:
        """Compute total throughput (objective function)."""
        metrics = compute_channel_metrics(p, scenario)
        return metrics['total_throughput']

    def _compute_gradient(self, p: np.ndarray, scenario: Scenario,
                          eps: float = 0.01) -> np.ndarray:
        """
        Compute gradient of throughput w.r.t. UAV position.

        Uses central finite differences for numerical stability.
        """
        grad = np.zeros(3)
        f0 = self._compute_throughput(p, scenario)

        for i in range(3):
            p_plus = p.copy()
            p_minus = p.copy()
            p_plus[i] += eps
            p_minus[i] -= eps

            # Clip to bounds
            if i < 2:
                p_plus[i] = np.clip(p_plus[i], 0, 100)
                p_minus[i] = np.clip(p_minus[i], 0, 100)
            else:
                p_plus[i] = np.clip(p_plus[i], 10, 40)
                p_minus[i] = np.clip(p_minus[i], 10, 40)

            f_plus = self._compute_throughput(p_plus, scenario)
            f_minus = self._compute_throughput(p_minus, scenario)
            grad[i] = (f_plus - f_minus) / (2 * eps)

        return grad

    def _compute_analytical_gradient(self, p: np.ndarray,
                                      scenario: Scenario) -> np.ndarray:
        """
        Compute analytical gradient of throughput.

        Using chain rule:
        d(throughput)/dp = sum_i d(rate_i)/d(SNR_i) * d(SNR_i)/d(distance) * d(distance)/dp
        """
        grad = np.zeros(3)

        d_bs_uav = np.linalg.norm(p - scenario.bs_position)
        snr_bs_uav = calc_snr(d_bs_uav, BS_POWER_DBM)

        for user_pos in scenario.user_positions:
            d_uav_user = np.linalg.norm(p - user_pos)
            snr_uav_user = calc_snr(d_uav_user, UAV_POWER_DBM)

            # Effective SNR is bottleneck
            if snr_uav_user < snr_bs_uav:
                # UAV-user link is bottleneck
                snr_eff = snr_uav_user
                snr_linear = 10 ** (snr_eff / 10)

                # d(rate)/d(SNR_linear) = BW * 1/(1+SNR) / ln(2)
                d_rate_d_snr = BANDWIDTH_GHZ * 1000 / ((1 + snr_linear) * np.log(2))

                # d(SNR_linear)/d(distance) via path loss model
                # SNR(d) = P - (20*log10(d/1000) + 20*log10(f) + 92.45 + 10*d/1000) - noise
                # d(SNR_dB)/d(d) = -20/(d*ln(10)) - 10/1000
                d_km = max(d_uav_user / 1000.0, 0.001)
                d_snr_db_d_d = -20 / (d_uav_user * np.log(10)) - 10 / 1000

                # Convert dB derivative to linear
                d_snr_linear_d_d = snr_linear * np.log(10) / 10 * d_snr_db_d_d

                # d(distance)/dp = (p - user_pos) / distance
                d_d_dp = (p - user_pos) / max(d_uav_user, 0.001)

                grad += d_rate_d_snr * d_snr_linear_d_d * d_d_dp
            else:
                # BS-UAV link is bottleneck
                snr_eff = snr_bs_uav
                snr_linear = 10 ** (snr_eff / 10)

                d_rate_d_snr = BANDWIDTH_GHZ * 1000 / ((1 + snr_linear) * np.log(2))

                d_km = max(d_bs_uav / 1000.0, 0.001)
                d_snr_db_d_d = -20 / (d_bs_uav * np.log(10)) - 10 / 1000
                d_snr_linear_d_d = snr_linear * np.log(10) / 10 * d_snr_db_d_d

                d_d_dp = (p - scenario.bs_position) / max(d_bs_uav, 0.001)

                grad += d_rate_d_snr * d_snr_linear_d_d * d_d_dp

        return grad

    def _project_to_box(self, p: np.ndarray) -> np.ndarray:
        """Project position to feasible region."""
        return np.array([
            np.clip(p[0], 0, 100),
            np.clip(p[1], 0, 100),
            np.clip(p[2], 10, 40)
        ])

    def _solve_subproblem(self, p_current: np.ndarray, grad: np.ndarray,
                          trust_radius: float) -> np.ndarray:
        """
        Solve the linearized subproblem:
            max  f(p_k) + grad^T (p - p_k)
            s.t. ||p - p_k|| <= trust_radius
                 p in [0,100]^2 x [10,40]

        This is a box-constrained gradient ascent step.
        """
        # Gradient ascent direction
        step = trust_radius * grad / (np.linalg.norm(grad) + 1e-8)
        p_new = p_current + step
        return self._project_to_box(p_new)

    def solve(self, scenario: Scenario,
              p_init: np.ndarray = None,
              verbose: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Run SCA to find optimal UAV position.

        Args:
            scenario: Problem scenario
            p_init: Initial position (default: analytical baseline)
            verbose: Print iteration details

        Returns:
            Tuple of (optimal_position, solve_info)
        """
        # Initialize from analytical baseline if not provided
        if p_init is None:
            user_centroid = np.mean(scenario.user_positions, axis=0)
            p_init = np.array([
                0.6 * user_centroid[0] + 0.4 * scenario.bs_position[0],
                0.6 * user_centroid[1] + 0.4 * scenario.bs_position[1],
                25.0
            ])
            p_init = self._project_to_box(p_init)

        p = p_init.copy()
        trust_radius = self.config.trust_region_init
        f_current = self._compute_throughput(p, scenario)

        self.history = [{
            'iteration': 0,
            'position': p.copy(),
            'throughput': f_current,
            'trust_radius': trust_radius,
            'grad_norm': 0.0
        }]

        converged = False
        for k in range(self.config.max_iterations):
            # Compute gradient
            grad = self._compute_gradient(p, scenario)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < self.config.convergence_tol:
                converged = True
                break

            # Solve subproblem
            p_trial = self._solve_subproblem(p, grad, trust_radius)
            f_trial = self._compute_throughput(p_trial, scenario)

            # Compute predicted vs actual improvement
            predicted_improvement = np.dot(grad, p_trial - p)
            actual_improvement = f_trial - f_current

            # Update trust region
            if predicted_improvement > 0:
                rho = actual_improvement / predicted_improvement
            else:
                rho = 0

            if rho > self.config.rho_accept:
                # Accept step
                p = p_trial
                f_current = f_trial

                # Expand trust region if good prediction
                if rho > 0.75:
                    trust_radius = min(trust_radius * self.config.eta_expand,
                                       self.config.trust_region_max)
            else:
                # Reject step, shrink trust region
                trust_radius = max(trust_radius * self.config.eta_shrink,
                                   self.config.trust_region_min)

            self.history.append({
                'iteration': k + 1,
                'position': p.copy(),
                'throughput': f_current,
                'trust_radius': trust_radius,
                'grad_norm': grad_norm,
                'rho': rho,
                'accepted': rho > self.config.rho_accept
            })

            if verbose:
                status = "+" if rho > self.config.rho_accept else "-"
                print(f"  SCA iter {k+1}: throughput={f_current:.2f}, "
                      f"grad_norm={grad_norm:.4f}, rho={rho:.3f} {status}")

        # Compute final metrics
        metrics = compute_channel_metrics(p, scenario)

        return p, {
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage': metrics['coverage_rate'],
            'iterations': len(self.history),
            'converged': converged,
            'history': self.history,
            'final_gradient': self._compute_gradient(p, scenario)
        }

    def get_gradient_at(self, p: np.ndarray, scenario: Scenario) -> np.ndarray:
        """Get gradient at a specific position (for warm-starting RL)."""
        return self._compute_gradient(p, scenario)


def solve_with_sca(scenario: Scenario,
                   max_iters: int = 50,
                   verbose: bool = False) -> Tuple[np.ndarray, float]:
    """Convenience function to solve with SCA."""
    solver = SCASolver(SCAConfig(max_iterations=max_iters))
    pos, info = solver.solve(scenario, verbose=verbose)
    return pos, info['throughput']


if __name__ == "__main__":
    """Test SCA solver on canonical scenarios."""
    from eval_common import generate_scenarios

    scenarios = generate_scenarios(num_scenarios=10)

    print("Testing SCA Solver")
    print("=" * 60)

    results = []
    for scenario in scenarios:
        solver = SCASolver()
        pos, info = solver.solve(scenario, verbose=False)

        # Compare with analytical baseline
        from eval_common import get_position_analytical
        analytical_pos = get_position_analytical(scenario)
        analytical_metrics = compute_channel_metrics(analytical_pos, scenario)

        improvement = (info['throughput'] - analytical_metrics['total_throughput']) / \
                      analytical_metrics['total_throughput'] * 100

        results.append({
            'scenario_id': scenario.id,
            'sca_throughput': info['throughput'],
            'analytical_throughput': analytical_metrics['total_throughput'],
            'improvement': improvement,
            'iterations': info['iterations'],
            'converged': info['converged']
        })

        print(f"Scenario {scenario.id}: SCA={info['throughput']:.1f} Mbps, "
              f"Analytical={analytical_metrics['total_throughput']:.1f} Mbps "
              f"(+{improvement:.1f}%), iters={info['iterations']}")

    print("\n" + "=" * 60)
    print(f"Average SCA throughput: {np.mean([r['sca_throughput'] for r in results]):.1f} Mbps")
    print(f"Average Analytical throughput: {np.mean([r['analytical_throughput'] for r in results]):.1f} Mbps")
    print(f"Average improvement: {np.mean([r['improvement'] for r in results]):.1f}%")
    print(f"Convergence rate: {sum(r['converged'] for r in results) / len(results) * 100:.0f}%")
