#!/usr/bin/env python3
"""
Shared utilities for VLA-6G evaluation scripts.

Extracted from evaluate_system.py to ensure identical scenario generation,
channel model, and baseline methods across all experiments (M1-M4).
"""

import numpy as np
import random
import json
import os
import re
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Test scenario configuration."""
    id: int
    num_users: int
    user_positions: List[np.ndarray]
    user_requirements: List[float]
    bs_position: np.ndarray
    initial_uav_position: np.ndarray


# ---------------------------------------------------------------------------
# Channel model  (identical to evaluate_system.py and channel_optimizer.py)
# ---------------------------------------------------------------------------

FREQUENCY_GHZ = 300.0
BANDWIDTH_GHZ = 10.0
BS_POWER_DBM = 30.0
UAV_POWER_DBM = 20.0


def calc_snr(distance: float, power_dbm: float) -> float:
    """Calculate SNR (dB) for a single link."""
    d_km = max(distance / 1000.0, 0.001)
    fspl = 20 * np.log10(d_km) + 20 * np.log10(FREQUENCY_GHZ) + 92.45
    absorption = 10.0 * d_km
    path_loss = fspl + absorption
    noise = -174 + 10 * np.log10(BANDWIDTH_GHZ * 1e9) + 10
    return power_dbm - path_loss - noise


def compute_channel_metrics(uav_pos: np.ndarray,
                            scenario: Scenario) -> Dict:
    """Compute channel metrics for a given UAV position.

    Returns dict with: total_throughput, average_rate, min_rate,
    fairness, coverage_rate, user_rates
    """
    d_bs_uav = np.linalg.norm(uav_pos - scenario.bs_position)
    snr_bs_uav = calc_snr(d_bs_uav, BS_POWER_DBM)

    user_rates = []
    users_covered = 0

    for i, user_pos in enumerate(scenario.user_positions):
        d_uav_user = np.linalg.norm(uav_pos - user_pos)
        snr_uav_user = calc_snr(d_uav_user, UAV_POWER_DBM)
        effective_snr = min(snr_bs_uav, snr_uav_user)

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
        'average_rate': np.mean(user_rates),
        'min_rate': min(user_rates),
        'fairness': fairness,
        'coverage_rate': users_covered / len(user_rates),
        'user_rates': user_rates,
    }


def compute_channel_metrics_raw(uav_pos: np.ndarray,
                                bs_position: np.ndarray,
                                user_positions: List[np.ndarray],
                                user_requirements: List[float]) -> Dict:
    """Channel metrics from raw arrays (no Scenario object needed)."""
    s = Scenario(id=0, num_users=len(user_positions),
                 user_positions=user_positions,
                 user_requirements=user_requirements,
                 bs_position=bs_position,
                 initial_uav_position=uav_pos)
    return compute_channel_metrics(uav_pos, s)


# ---------------------------------------------------------------------------
# Composite score (same weights as channel_optimizer.py)
# ---------------------------------------------------------------------------

def composite_score(metrics: Dict,
                    weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)) -> float:
    """Weighted composite: 0.6*throughput + 0.3*fairness*1000 + 0.1*coverage*1000."""
    w1, w2, w3 = weights
    return (w1 * metrics['total_throughput']
            + w2 * metrics['fairness'] * 1000
            + w3 * metrics['coverage_rate'] * 1000)


# ---------------------------------------------------------------------------
# Scenario generation (identical to evaluate_system.py, seed=42)
# ---------------------------------------------------------------------------

def generate_scenarios(num_scenarios: int = 100,
                       seed: int = 42,
                       bs_position: np.ndarray = None) -> List[Scenario]:
    """Generate the canonical set of evaluation scenarios."""
    if bs_position is None:
        bs_position = np.array([0.0, 0.0, 30.0])

    scenarios = []
    random.seed(seed)

    for i in range(num_scenarios):
        num_users = random.choice([3, 4, 5, 6, 7])

        if i % 4 == 0:
            # Clustered
            center = np.array([random.uniform(40, 60), random.uniform(40, 60)])
            user_positions = [
                np.array([
                    center[0] + random.gauss(0, 5),
                    center[1] + random.gauss(0, 5),
                    1.0
                ])
                for _ in range(num_users)
            ]
        elif i % 4 == 1:
            # Spread
            user_positions = [
                np.array([
                    random.uniform(20, 80),
                    random.uniform(20, 80),
                    1.0
                ])
                for _ in range(num_users)
            ]
        elif i % 4 == 2:
            # Line
            y_pos = random.uniform(30, 70)
            user_positions = [
                np.array([20 + j * 60 / (num_users - 1), y_pos, 1.0])
                for j in range(num_users)
            ]
        else:
            # Circle
            center = np.array([50, 50])
            radius = random.uniform(15, 30)
            user_positions = [
                np.array([
                    center[0] + radius * np.cos(2 * np.pi * j / num_users),
                    center[1] + radius * np.sin(2 * np.pi * j / num_users),
                    1.0
                ])
                for j in range(num_users)
            ]

        user_requirements = [
            random.uniform(10, 50) for _ in range(num_users)
        ]

        initial_uav = np.array([
            random.uniform(30, 70),
            random.uniform(30, 70),
            random.uniform(15, 35)
        ])

        scenarios.append(Scenario(
            id=i,
            num_users=num_users,
            user_positions=user_positions,
            user_requirements=user_requirements,
            bs_position=bs_position,
            initial_uav_position=initial_uav,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# Baseline positioning methods
# ---------------------------------------------------------------------------

def get_position_analytical(scenario: Scenario) -> np.ndarray:
    """Weighted centroid baseline."""
    user_centroid = np.mean(scenario.user_positions, axis=0)
    bs_2d = scenario.bs_position[:2]
    centroid_2d = user_centroid[:2]

    alpha = 0.6
    optimal_xy = (1 - alpha) * bs_2d + alpha * centroid_2d

    distances = [np.linalg.norm(optimal_xy - u[:2]) for u in scenario.user_positions]
    avg_dist = np.mean(distances)
    optimal_z = np.clip(0.5 * avg_dist, 10.0, 40.0)

    return np.array([optimal_xy[0], optimal_xy[1], optimal_z])


def get_position_random(scenario: Scenario) -> np.ndarray:
    """Random position within operational area."""
    return np.array([
        random.uniform(20, 80),
        random.uniform(20, 80),
        random.uniform(10, 40)
    ])


def get_position_static(scenario: Scenario) -> np.ndarray:
    """Static center position."""
    return np.array([50.0, 50.0, 25.0])


# ---------------------------------------------------------------------------
# VLA prompt / parse helpers (for reuse in benchmark_latency and mobility)
# ---------------------------------------------------------------------------

def format_vla_prompt(scenario: Scenario) -> str:
    """Build the VLA inference prompt (matches training format)."""
    prompt = f"""You are a UAV relay positioning expert for 6G networks.

Current situation:
- Base station at: ({scenario.bs_position[0]:.1f}, {scenario.bs_position[1]:.1f}, {scenario.bs_position[2]:.1f})
- UAV currently at: ({scenario.initial_uav_position[0]:.1f}, {scenario.initial_uav_position[1]:.1f}, {scenario.initial_uav_position[2]:.1f})
- Number of ground users: {scenario.num_users}
- Current total throughput: 0.0 Mbps
- Current fairness index: 0.000

User details:
"""
    for i, user_pos in enumerate(scenario.user_positions):
        req = scenario.user_requirements[i]
        prompt += f"  User {i}: pos=({user_pos[0]:.1f}, {user_pos[1]:.1f}), "
        prompt += f"SNR=0.0dB, rate=0.0Mbps, "
        prompt += f"required={req:.1f}Mbps, covered=False\n"

    prompt += """
Task: Determine the optimal UAV relay position.
Output ONLY valid JSON: {"x": float, "y": float, "z": float}
"""
    return prompt


def parse_vla_response(response: str, scenario: Scenario) -> Tuple[np.ndarray, bool]:
    """Extract x,y,z from model output. Returns (position, parsed_ok)."""
    try:
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if 'x' in data and 'y' in data and 'z' in data:
                    x, y, z = float(data['x']), float(data['y']), float(data['z'])
                    return np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)]), True
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        simple_pattern = r'"x":\s*([\d.]+)\s*,\s*"y":\s*([\d.]+)\s*,\s*"z":\s*([\d.]+)'
        match = re.search(simple_pattern, response, re.IGNORECASE)
        if match:
            x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
            return np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)]), True

        tuple_pattern = r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)'
        match = re.search(tuple_pattern, response)
        if match:
            x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
            return np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)]), True

        numbers = re.findall(r'(\d+\.?\d*)', response)
        if len(numbers) >= 3:
            x, y, z = float(numbers[0]), float(numbers[1]), float(numbers[2])
            if 0 <= x <= 100 and 0 <= y <= 100 and 1 <= z <= 60:
                return np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)]), True
    except Exception:
        pass

    return get_position_analytical(scenario), False


# ---------------------------------------------------------------------------
# Feature extraction (for MLP and DRL)
# ---------------------------------------------------------------------------

MAX_USERS = 7  # Pad/truncate to this many users


def extract_features(scenario: Scenario) -> np.ndarray:
    """Extract a fixed-size 37-dim feature vector from a scenario.

    Layout: bs_pos(3) + uav_pos(3) + user_xy(7*2=14, zero-padded)
            + num_users(1) + user_snrs(7, zero-padded)
            + user_rates(7, zero-padded) + throughput(1) + fairness(1)
    Total = 3 + 3 + 14 + 1 + 7 + 7 + 1 + 1 = 37
    """
    # Compute metrics at initial UAV position for feature enrichment
    metrics = compute_channel_metrics(scenario.initial_uav_position, scenario)

    bs = scenario.bs_position
    uav = scenario.initial_uav_position

    # User XY positions (padded to MAX_USERS)
    user_xy = np.zeros(MAX_USERS * 2)
    for j, upos in enumerate(scenario.user_positions[:MAX_USERS]):
        user_xy[2 * j] = upos[0]
        user_xy[2 * j + 1] = upos[1]

    # Number of users (normalized)
    n_users = scenario.num_users

    # Per-user SNR (padded)
    user_snrs = np.zeros(MAX_USERS)
    d_bs_uav = np.linalg.norm(uav - bs)
    snr_bs_uav = calc_snr(d_bs_uav, BS_POWER_DBM)
    for j, upos in enumerate(scenario.user_positions[:MAX_USERS]):
        d_uav_user = np.linalg.norm(uav - upos)
        user_snrs[j] = min(snr_bs_uav, calc_snr(d_uav_user, UAV_POWER_DBM))

    # Per-user rates (padded)
    user_rates = np.zeros(MAX_USERS)
    for j, r in enumerate(metrics['user_rates'][:MAX_USERS]):
        user_rates[j] = r

    features = np.concatenate([
        bs,                                    # 3
        uav,                                   # 3
        user_xy,                               # 14
        [float(n_users)],                      # 1
        user_snrs,                             # 7
        user_rates,                            # 7
        [metrics['total_throughput']],         # 1
        [metrics['fairness']],                 # 1
    ])
    return features.astype(np.float32)


def extract_features_from_raw(bs_position: np.ndarray,
                              uav_position: np.ndarray,
                              user_positions: List[np.ndarray],
                              user_requirements: List[float],
                              user_snrs: List[float],
                              user_rates: List[float],
                              throughput: float,
                              fairness: float) -> np.ndarray:
    """Extract features from raw training sample data (no Scenario needed)."""
    user_xy = np.zeros(MAX_USERS * 2)
    for j, upos in enumerate(user_positions[:MAX_USERS]):
        user_xy[2 * j] = upos[0]
        user_xy[2 * j + 1] = upos[1]

    snrs_padded = np.zeros(MAX_USERS)
    for j, s in enumerate(user_snrs[:MAX_USERS]):
        snrs_padded[j] = s

    rates_padded = np.zeros(MAX_USERS)
    for j, r in enumerate(user_rates[:MAX_USERS]):
        rates_padded[j] = r

    features = np.concatenate([
        bs_position,
        uav_position,
        user_xy,
        [float(len(user_positions))],
        snrs_padded,
        rates_padded,
        [throughput],
        [fairness],
    ])
    return features.astype(np.float32)


FEATURE_DIM = 37  # for convenience
FEATURE_DIM_OBSTACLE_SIMPLE = 40  # 37 + 3 binary obstacle flags
FEATURE_DIM_OBSTACLE_RICH = 51  # 37 + 14 rich obstacle features
MAX_OBSTACLES = 3  # Maximum number of obstacles to encode


# ---------------------------------------------------------------------------
# Obstacle data structures and channel model
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """Axis-aligned bounding box obstacle (building)."""
    min_corner: np.ndarray  # [x_min, y_min, z_min]
    max_corner: np.ndarray  # [x_max, y_max, z_max]

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the obstacle."""
        return all(self.min_corner <= point) and all(point <= self.max_corner)


@dataclass
class ObstacleScenario:
    """Test scenario with obstacles."""
    id: int
    num_users: int
    user_positions: List[np.ndarray]
    user_requirements: List[float]
    bs_position: np.ndarray
    initial_uav_position: np.ndarray
    obstacles: List[Obstacle]  # List of building obstacles


NLOS_PENALTY_DB = 20.0  # dB penalty when LoS is blocked


def line_intersects_box(p1: np.ndarray, p2: np.ndarray,
                        box_min: np.ndarray, box_max: np.ndarray) -> bool:
    """Check if line segment p1->p2 intersects axis-aligned bounding box.

    Uses slab method for 3D ray-box intersection.
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return False

    direction = direction / length

    # Handle division by zero for axis-aligned rays
    inv_dir = np.zeros(3)
    for i in range(3):
        if abs(direction[i]) < 1e-10:
            inv_dir[i] = 1e10 if direction[i] >= 0 else -1e10
        else:
            inv_dir[i] = 1.0 / direction[i]

    t_min = -np.inf
    t_max = np.inf

    for i in range(3):
        t1 = (box_min[i] - p1[i]) * inv_dir[i]
        t2 = (box_max[i] - p1[i]) * inv_dir[i]

        if t1 > t2:
            t1, t2 = t2, t1

        t_min = max(t_min, t1)
        t_max = min(t_max, t2)

        if t_min > t_max:
            return False

    # Check if intersection is within segment length
    return t_min <= length and t_max >= 0


def check_los_blocked(tx_pos: np.ndarray, rx_pos: np.ndarray,
                      obstacles: List[Obstacle]) -> bool:
    """Check if line-of-sight between tx and rx is blocked by any obstacle."""
    for obs in obstacles:
        if line_intersects_box(tx_pos, rx_pos, obs.min_corner, obs.max_corner):
            return True
    return False


def calc_snr_with_obstacles(distance: float, power_dbm: float,
                            los_blocked: bool) -> float:
    """Calculate SNR with optional NLoS penalty when LoS is blocked."""
    snr = calc_snr(distance, power_dbm)
    if los_blocked:
        snr -= NLOS_PENALTY_DB
    return snr


def compute_channel_metrics_obstacles(uav_pos: np.ndarray,
                                       scenario: ObstacleScenario) -> Dict:
    """Compute channel metrics considering obstacle blockage.

    Same as compute_channel_metrics but applies 20 dB NLoS penalty
    when LoS is blocked by obstacles.
    """
    d_bs_uav = np.linalg.norm(uav_pos - scenario.bs_position)

    # Check BS-UAV link blockage
    bs_uav_blocked = check_los_blocked(scenario.bs_position, uav_pos,
                                        scenario.obstacles)
    snr_bs_uav = calc_snr_with_obstacles(d_bs_uav, BS_POWER_DBM, bs_uav_blocked)

    user_rates = []
    users_covered = 0

    for i, user_pos in enumerate(scenario.user_positions):
        d_uav_user = np.linalg.norm(uav_pos - user_pos)

        # Check UAV-user link blockage
        uav_user_blocked = check_los_blocked(uav_pos, user_pos,
                                              scenario.obstacles)
        snr_uav_user = calc_snr_with_obstacles(d_uav_user, UAV_POWER_DBM,
                                                uav_user_blocked)

        effective_snr = min(snr_bs_uav, snr_uav_user)

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
        'average_rate': np.mean(user_rates),
        'min_rate': min(user_rates),
        'fairness': fairness,
        'coverage_rate': users_covered / len(user_rates),
        'user_rates': user_rates,
        'bs_uav_blocked': bs_uav_blocked,
        'num_blocked_users': sum(1 for i, u in enumerate(scenario.user_positions)
                                  if check_los_blocked(uav_pos, u, scenario.obstacles)),
    }


def generate_obstacle_scenarios(num_scenarios: int = 50,
                                 seed: int = 42,
                                 bs_position: np.ndarray = None,
                                 num_obstacles_range: Tuple[int, int] = (1, 3)) -> List[ObstacleScenario]:
    """Generate evaluation scenarios with random building obstacles.

    Creates realistic urban scenarios with 1-3 buildings that may block
    LoS between UAV relay and users/base station.
    """
    if bs_position is None:
        bs_position = np.array([0.0, 0.0, 30.0])

    scenarios = []
    random.seed(seed)
    np.random.seed(seed)

    for i in range(num_scenarios):
        num_users = random.choice([3, 4, 5, 6, 7])

        # Generate obstacles first
        num_obstacles = random.randint(*num_obstacles_range)
        obstacles = []

        for _ in range(num_obstacles):
            # Random building footprint
            width = random.uniform(8, 20)  # 8-20m wide
            depth = random.uniform(8, 20)  # 8-20m deep
            height = random.uniform(15, 35)  # 15-35m tall

            # Random position (avoid area edges)
            x_center = random.uniform(25, 75)
            y_center = random.uniform(25, 75)

            min_corner = np.array([
                x_center - width/2,
                y_center - depth/2,
                0.0
            ])
            max_corner = np.array([
                x_center + width/2,
                y_center + depth/2,
                height
            ])

            obstacles.append(Obstacle(min_corner=min_corner, max_corner=max_corner))

        # Generate user positions (avoiding obstacles)
        user_positions = []
        attempts = 0
        while len(user_positions) < num_users and attempts < 100:
            if i % 4 == 0:
                # Clustered
                center = np.array([random.uniform(40, 60), random.uniform(40, 60)])
                pos = np.array([
                    center[0] + random.gauss(0, 5),
                    center[1] + random.gauss(0, 5),
                    1.0
                ])
            elif i % 4 == 1:
                # Spread
                pos = np.array([
                    random.uniform(20, 80),
                    random.uniform(20, 80),
                    1.0
                ])
            elif i % 4 == 2:
                # Linear
                y_pos = random.uniform(30, 70)
                idx = len(user_positions)
                pos = np.array([20 + idx * 60 / max(num_users - 1, 1), y_pos, 1.0])
            else:
                # Circular
                center = np.array([50, 50])
                radius = random.uniform(15, 30)
                idx = len(user_positions)
                pos = np.array([
                    center[0] + radius * np.cos(2 * np.pi * idx / num_users),
                    center[1] + radius * np.sin(2 * np.pi * idx / num_users),
                    1.0
                ])

            # Check if position is inside any obstacle
            inside_obstacle = False
            for obs in obstacles:
                if obs.contains_point(pos):
                    inside_obstacle = True
                    break

            if not inside_obstacle:
                user_positions.append(pos)
            attempts += 1

        # Fill remaining users if couldn't place all
        while len(user_positions) < num_users:
            user_positions.append(np.array([50.0, 50.0, 1.0]))

        user_requirements = [random.uniform(10, 50) for _ in range(num_users)]

        initial_uav = np.array([
            random.uniform(30, 70),
            random.uniform(30, 70),
            random.uniform(15, 35)
        ])

        scenarios.append(ObstacleScenario(
            id=i,
            num_users=num_users,
            user_positions=user_positions,
            user_requirements=user_requirements,
            bs_position=bs_position,
            initial_uav_position=initial_uav,
            obstacles=obstacles,
        ))

    return scenarios


def get_obstacle_flags(scenario: ObstacleScenario, uav_pos: np.ndarray = None) -> np.ndarray:
    """Extract 3 binary obstacle flags for MLP input.

    Flags indicate presence of obstacles in 3 sectors:
    - Sector 0: Between BS and area center (x < 50)
    - Sector 1: Area center (25 < x < 75, 25 < y < 75)
    - Sector 2: Far side from BS (x > 50)
    """
    flags = np.zeros(3, dtype=np.float32)

    for obs in scenario.obstacles:
        center = (obs.min_corner + obs.max_corner) / 2

        # Sector 0: Left side (closer to BS at x=0)
        if center[0] < 40:
            flags[0] = 1.0
        # Sector 1: Center area
        elif 30 < center[0] < 70 and 30 < center[1] < 70:
            flags[1] = 1.0
        # Sector 2: Right side (far from BS)
        else:
            flags[2] = 1.0

    return flags


def extract_features_obstacle(scenario: ObstacleScenario) -> np.ndarray:
    """Extract 40-dim feature vector for obstacle-aware MLP (simple version).

    Layout: standard 37 features + 3 obstacle sector flags = 40
    """
    # Create a regular Scenario for base feature extraction
    base_scenario = Scenario(
        id=scenario.id,
        num_users=scenario.num_users,
        user_positions=scenario.user_positions,
        user_requirements=scenario.user_requirements,
        bs_position=scenario.bs_position,
        initial_uav_position=scenario.initial_uav_position,
    )

    base_features = extract_features(base_scenario)
    obstacle_flags = get_obstacle_flags(scenario)

    return np.concatenate([base_features, obstacle_flags]).astype(np.float32)


def extract_features_obstacle_rich(scenario: ObstacleScenario) -> np.ndarray:
    """Extract 51-dim feature vector with rich obstacle information.

    Layout: standard 37 features + 14 obstacle features = 51
    Obstacle features (14 total):
    - num_obstacles (1): normalized count
    - obstacle_info (9): 3 obstacles × 3 features (center_x, center_y, height) normalized
    - user_blockage (4): aggregated blockage indicators (padded to 4)
      - fraction of users with blocked LoS to center
      - fraction of users with blocked LoS to BS
      - max obstacle height
      - obstacle coverage area fraction
    """
    # Create a regular Scenario for base feature extraction
    base_scenario = Scenario(
        id=scenario.id,
        num_users=scenario.num_users,
        user_positions=scenario.user_positions,
        user_requirements=scenario.user_requirements,
        bs_position=scenario.bs_position,
        initial_uav_position=scenario.initial_uav_position,
    )

    base_features = extract_features(base_scenario)

    # Obstacle features
    obstacle_features = np.zeros(14, dtype=np.float32)

    # Number of obstacles (normalized by max)
    obstacle_features[0] = len(scenario.obstacles) / MAX_OBSTACLES

    # Per-obstacle info (up to 3 obstacles)
    for i, obs in enumerate(scenario.obstacles[:MAX_OBSTACLES]):
        center = (obs.min_corner + obs.max_corner) / 2
        height = obs.max_corner[2]
        # Normalized positions and height
        obstacle_features[1 + i*3] = center[0] / 100.0  # x normalized
        obstacle_features[2 + i*3] = center[1] / 100.0  # y normalized
        obstacle_features[3 + i*3] = height / 40.0  # height normalized

    # Aggregated blockage indicators
    area_center = np.array([50.0, 50.0, 25.0])  # Typical UAV position

    # Fraction of users with potential LoS blockage to area center
    blocked_to_center = 0
    blocked_to_bs = 0
    for user_pos in scenario.user_positions:
        if check_los_blocked(user_pos, area_center, scenario.obstacles):
            blocked_to_center += 1
        if check_los_blocked(user_pos, scenario.bs_position, scenario.obstacles):
            blocked_to_bs += 1

    obstacle_features[10] = blocked_to_center / len(scenario.user_positions)
    obstacle_features[11] = blocked_to_bs / len(scenario.user_positions)

    # Max obstacle height (normalized)
    if scenario.obstacles:
        max_height = max(obs.max_corner[2] for obs in scenario.obstacles)
        obstacle_features[12] = max_height / 40.0
    else:
        obstacle_features[12] = 0.0

    # Total obstacle coverage area (fraction of 100×100 area)
    total_area = 0.0
    for obs in scenario.obstacles:
        width = obs.max_corner[0] - obs.min_corner[0]
        depth = obs.max_corner[1] - obs.min_corner[1]
        total_area += width * depth
    obstacle_features[13] = min(total_area / 10000.0, 1.0)

    return np.concatenate([base_features, obstacle_features]).astype(np.float32)


def format_vla_prompt_obstacles(scenario: ObstacleScenario) -> str:
    """Build VLA inference prompt with obstacle information."""
    prompt = f"""You are a UAV relay positioning expert for 6G networks.

Current situation:
- Base station at: ({scenario.bs_position[0]:.1f}, {scenario.bs_position[1]:.1f}, {scenario.bs_position[2]:.1f})
- UAV currently at: ({scenario.initial_uav_position[0]:.1f}, {scenario.initial_uav_position[1]:.1f}, {scenario.initial_uav_position[2]:.1f})
- Number of ground users: {scenario.num_users}
- Current total throughput: 0.0 Mbps
- Current fairness index: 0.000

Building obstructions:
"""
    for i, obs in enumerate(scenario.obstacles):
        center = (obs.min_corner + obs.max_corner) / 2
        size = obs.max_corner - obs.min_corner
        prompt += f"  Building {i+1}: center=({center[0]:.1f}, {center[1]:.1f}), "
        prompt += f"size=({size[0]:.1f}x{size[1]:.1f}), height={obs.max_corner[2]:.1f}m\n"

    prompt += "\nUser details:\n"
    for i, user_pos in enumerate(scenario.user_positions):
        req = scenario.user_requirements[i]
        prompt += f"  User {i}: pos=({user_pos[0]:.1f}, {user_pos[1]:.1f}), "
        prompt += f"SNR=0.0dB, rate=0.0Mbps, "
        prompt += f"required={req:.1f}Mbps, covered=False\n"

    prompt += """
Task: Determine the optimal UAV relay position to avoid building obstructions.
Output ONLY valid JSON: {"x": float, "y": float, "z": float}
"""
    return prompt


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

RESULTS_DIR = '/home/it-services/ros2_ws/src/vla_6g_tvt/results'
FIGURES_DIR = '/home/it-services/ros2_ws/src/vla_6g_tvt/paper/figures'
DATA_DIR = '/home/it-services/ros2_ws/src/vla_6g_tvt/data'
MODEL_DIR = '/home/it-services/ros2_ws/src/vla_6g_tvt/models/vla_6g_v1'


def save_results(data: dict, prefix: str) -> str:
    """Save results JSON with timestamp, return path."""
    from datetime import datetime
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'{prefix}_{ts}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {path}")
    return path
