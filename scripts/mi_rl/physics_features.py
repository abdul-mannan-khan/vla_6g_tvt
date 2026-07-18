#!/usr/bin/env python3
"""
Physics-Informed Feature Engineering for UAV Relay Positioning.

Instead of learning to approximate physics from scratch, we encode
domain knowledge directly into features:
1. Channel-aware distances (not just Euclidean)
2. Gradient information from analytical model
3. Bottleneck identification
4. Spatial relationships with physical meaning

This gives the RL agent a "head start" over pure learning.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_common import (
    Scenario, compute_channel_metrics, calc_snr,
    FREQUENCY_GHZ, BANDWIDTH_GHZ, BS_POWER_DBM, UAV_POWER_DBM, MAX_USERS
)
from classical.analytical_gradients import (
    compute_throughput_gradient, compute_snr_gradient, GradientOracle
)


def extract_physics_features(p: np.ndarray, scenario: Scenario) -> np.ndarray:
    """
    Extract physics-informed features for RL state.

    Features are designed to encode physical relationships that
    the channel model captures, reducing what the network must learn.

    Feature groups:
    1. Position features (normalized)
    2. Distance features (BS-UAV, UAV-users)
    3. SNR features (bottleneck, margins)
    4. Gradient features (optimal direction)
    5. Topology features (user distribution)

    Total: 47 dimensions
    """
    features = []

    # --- Group 1: Position features (6 dims) ---
    # Normalized positions [0, 1]
    pos_norm = p / np.array([100, 100, 40])
    bs_norm = scenario.bs_position / np.array([100, 100, 40])
    features.extend(pos_norm)
    features.extend(bs_norm)

    # --- Group 2: Distance features (10 dims) ---
    d_bs_uav = np.linalg.norm(p - scenario.bs_position)
    features.append(d_bs_uav / 100)  # Normalized

    user_distances = []
    for user_pos in scenario.user_positions[:MAX_USERS]:
        d = np.linalg.norm(p - user_pos)
        user_distances.append(d)

    # Pad to MAX_USERS
    while len(user_distances) < MAX_USERS:
        user_distances.append(0)

    # Statistics
    features.append(np.mean(user_distances) / 100)
    features.append(np.std(user_distances) / 100)
    features.append(np.min([d for d in user_distances if d > 0] or [0]) / 100)

    # --- Group 3: SNR features (10 dims) ---
    snr_bs_uav = calc_snr(d_bs_uav, BS_POWER_DBM)
    features.append(snr_bs_uav / 50)  # Normalized roughly to [-1, 1]

    user_snrs = []
    bottleneck_counts = {'bs_uav': 0, 'uav_user': 0}
    for user_pos in scenario.user_positions[:MAX_USERS]:
        d = np.linalg.norm(p - user_pos)
        snr_uav_user = calc_snr(d, UAV_POWER_DBM)
        user_snrs.append(snr_uav_user)

        if snr_uav_user < snr_bs_uav:
            bottleneck_counts['uav_user'] += 1
        else:
            bottleneck_counts['bs_uav'] += 1

    while len(user_snrs) < MAX_USERS:
        user_snrs.append(0)

    features.extend([s / 50 for s in user_snrs[:MAX_USERS]])

    # Bottleneck ratio
    total = bottleneck_counts['bs_uav'] + bottleneck_counts['uav_user']
    features.append(bottleneck_counts['uav_user'] / max(total, 1))

    # SNR margin (how close to switching bottleneck)
    if user_snrs:
        min_user_snr = min([s for s in user_snrs if s != 0] or [0])
        snr_margin = abs(snr_bs_uav - min_user_snr) / 50
        features.append(snr_margin)
    else:
        features.append(0)

    # --- Group 4: Gradient features (6 dims) ---
    try:
        throughput, grad = compute_throughput_gradient(p, scenario)
        grad_norm = np.linalg.norm(grad)

        # Normalized gradient direction
        if grad_norm > 1e-6:
            grad_dir = grad / grad_norm
        else:
            grad_dir = np.zeros(3)

        features.extend(grad_dir)
        features.append(grad_norm / 1000)  # Normalized gradient magnitude
        features.append(throughput / 500)  # Normalized throughput
        features.append(np.log1p(grad_norm) / 10)  # Log gradient magnitude
    except:
        features.extend([0, 0, 0, 0, 0, 0])

    # --- Group 5: Topology features (9 dims) ---
    # User centroid
    if scenario.user_positions:
        user_centroid = np.mean(scenario.user_positions, axis=0)[:2]
    else:
        user_centroid = np.array([50, 50])

    features.extend(user_centroid / 100)

    # Distance from centroid
    d_to_centroid = np.linalg.norm(p[:2] - user_centroid)
    features.append(d_to_centroid / 100)

    # User spread (std of positions)
    if len(scenario.user_positions) > 1:
        user_xy = np.array([u[:2] for u in scenario.user_positions])
        user_spread = np.std(user_xy)
    else:
        user_spread = 0
    features.append(user_spread / 50)

    # Number of users (normalized)
    features.append(scenario.num_users / MAX_USERS)

    # Optimal height heuristic
    optimal_z_heuristic = np.clip(0.5 * d_to_centroid, 10, 40)
    features.append((p[2] - optimal_z_heuristic) / 40)  # Height deviation

    # Angular coverage (how spread out users are from UAV perspective)
    angles = []
    for user_pos in scenario.user_positions[:MAX_USERS]:
        diff = user_pos[:2] - p[:2]
        angle = np.arctan2(diff[1], diff[0])
        angles.append(angle)
    if len(angles) > 1:
        angles = np.sort(angles)
        angular_gaps = np.diff(angles)
        angular_gaps = np.append(angular_gaps, 2*np.pi - (angles[-1] - angles[0]))
        max_angular_gap = np.max(angular_gaps)
        features.append(max_angular_gap / (2 * np.pi))
    else:
        features.append(0)

    # --- Group 6: Requirement features (6 dims) ---
    # User requirements relative to current rates
    metrics = compute_channel_metrics(p, scenario)
    rates = metrics['user_rates']
    reqs = scenario.user_requirements

    rate_margins = []
    for i in range(min(len(rates), MAX_USERS)):
        margin = rates[i] - reqs[i]
        rate_margins.append(margin / 100)

    while len(rate_margins) < 3:
        rate_margins.append(0)

    features.extend(rate_margins[:3])  # First 3 user margins
    features.append(metrics['coverage_rate'])
    features.append(metrics['fairness'])
    features.append(np.mean(rates) / 200 if rates else 0)

    return np.array(features, dtype=np.float32)


PHYSICS_FEATURE_DIM = 47


def extract_sca_warm_start(scenario: Scenario, p_current: np.ndarray = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get SCA-suggested position and gradient for warm-starting policy.

    Returns:
        sca_position: Suggested next position from SCA step
        gradient: Normalized gradient direction
    """
    from classical.sca_solver import SCASolver, SCAConfig

    if p_current is None:
        p_current = scenario.initial_uav_position

    # Single SCA iteration for warm start
    solver = SCASolver(SCAConfig(max_iterations=1))
    sca_pos, info = solver.solve(scenario, p_init=p_current)

    # Get gradient at current position
    _, grad = compute_throughput_gradient(p_current, scenario)
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-6:
        grad_dir = grad / grad_norm
    else:
        grad_dir = np.zeros(3)

    return sca_pos, grad_dir


def compute_physics_loss(predicted_pos: np.ndarray,
                         predicted_throughput: float,
                         scenario: Scenario) -> float:
    """
    Compute physics-informed loss for training.

    This loss penalizes predictions that violate channel physics:
    1. Predicted throughput should match channel model
    2. Position should be in feasible region

    Returns:
        physics_loss: Penalty for violating physics
    """
    # Compute actual throughput at predicted position
    metrics = compute_channel_metrics(predicted_pos, scenario)
    actual_throughput = metrics['total_throughput']

    # Throughput prediction error
    throughput_loss = (predicted_throughput - actual_throughput) ** 2 / 10000

    # Position feasibility (soft constraint)
    pos_penalty = 0.0
    pos_penalty += max(0, -predicted_pos[0]) ** 2  # x >= 0
    pos_penalty += max(0, predicted_pos[0] - 100) ** 2  # x <= 100
    pos_penalty += max(0, -predicted_pos[1]) ** 2  # y >= 0
    pos_penalty += max(0, predicted_pos[1] - 100) ** 2  # y <= 100
    pos_penalty += max(0, 10 - predicted_pos[2]) ** 2  # z >= 10
    pos_penalty += max(0, predicted_pos[2] - 40) ** 2  # z <= 40

    return throughput_loss + 0.1 * pos_penalty


if __name__ == "__main__":
    """Test physics features extraction."""
    from eval_common import generate_scenarios

    scenarios = generate_scenarios(num_scenarios=5)

    print("Testing Physics Features")
    print("=" * 60)

    for scenario in scenarios[:3]:
        p = scenario.initial_uav_position
        features = extract_physics_features(p, scenario)

        print(f"Scenario {scenario.id}:")
        print(f"  Position: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
        print(f"  Feature dimension: {len(features)}")
        print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Non-zero features: {np.sum(features != 0)}")

        # Test warm start
        sca_pos, grad_dir = extract_sca_warm_start(scenario, p)
        print(f"  SCA warm start: ({sca_pos[0]:.1f}, {sca_pos[1]:.1f}, {sca_pos[2]:.1f})")
        print(f"  Gradient direction: [{grad_dir[0]:.3f}, {grad_dir[1]:.3f}, {grad_dir[2]:.3f}]")
        print()
