"""
UAV Relay Positioning Environment for 6G IoT Networks
Implements the system model from the MI-RL paper.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class ChannelParams:
    """Channel and network parameters."""
    fc: float = 28e9           # Carrier frequency (Hz) - mmWave
    B: float = 100e6           # Bandwidth (Hz)
    P_bs: float = 1.0          # BS transmit power (W) = 30 dBm
    P_uav: float = 1.0         # UAV transmit power (W) = 30 dBm
    N0: float = 1e-20          # Noise PSD (W/Hz)
    c: float = 3e8             # Speed of light (m/s)

    # Environment geometry
    area_size: float = 100.0   # Area size (m)
    h_min: float = 10.0        # Min UAV altitude (m)
    h_max: float = 40.0        # Max UAV altitude (m)

    # Antenna gains (linear scale)
    G_bs: float = 10.0         # BS antenna gain
    G_uav: float = 5.0         # UAV antenna gain
    G_user: float = 1.0        # User antenna gain


class UAVRelayEnv:
    """
    UAV Relay Positioning Environment.

    The UAV acts as a decode-and-forward relay between a base station
    and multiple IoT ground users. The goal is to maximize total throughput.
    """

    def __init__(
        self,
        num_users: int = 5,
        params: Optional[ChannelParams] = None,
        max_steps: int = 50,
        seed: Optional[int] = None
    ):
        self.num_users = num_users
        self.params = params or ChannelParams()
        self.max_steps = max_steps

        if seed is not None:
            np.random.seed(seed)

        # State and action dimensions
        # State: [uav_pos(3), sca_pos(3), sca_grad(3), snr_bu(1), snr_users(K),
        #         throughput_current(1), throughput_sca(1)]
        self.state_dim = 3 + 3 + 3 + 1 + num_users + 1 + 1
        self.action_dim = 3  # Position correction (dx, dy, dh)

        # Fixed positions
        self.bs_pos = np.array([50.0, 50.0, 15.0])

        # Episode state
        self.user_positions = None
        self.uav_pos = None
        self.sca_pos = None
        self.sca_grad = None
        self.sca_throughput = None
        self.step_count = 0

    def reset(self, user_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment with new scenario.

        Args:
            user_positions: Optional fixed user positions (K x 3)

        Returns:
            Initial state vector
        """
        self.step_count = 0

        # Generate or use provided user positions
        if user_positions is not None:
            self.user_positions = user_positions.copy()
        else:
            self.user_positions = self._generate_random_users()

        # Compute SCA solution for this scenario
        self.sca_pos, self.sca_grad = self._run_sca(num_iters=20)
        self.sca_throughput = self._compute_throughput(self.sca_pos)

        # Initialize UAV at SCA position (warm start)
        self.uav_pos = self.sca_pos.copy()

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return next state, reward, done, info.

        Args:
            action: Position correction [dx, dy, dh]

        Returns:
            next_state, reward, done, info dict
        """
        self.step_count += 1

        # Apply action with step size
        step_size = 2.0
        new_pos = self.uav_pos + step_size * action

        # Project to feasible region
        new_pos = self._project_to_feasible(new_pos)

        # Compute throughput at new position
        new_throughput = self._compute_throughput(new_pos)

        # Performance floor guarantee
        floor_activated = False
        if new_throughput < self.sca_throughput:
            new_pos = self.sca_pos.copy()
            new_throughput = self.sca_throughput
            floor_activated = True

        # Compute reward
        improvement = new_throughput - self.sca_throughput

        # Normalize reward to reasonable scale
        reward = improvement / 1e6  # Scale to Mbps improvement

        # Small penalty for large deviations (encourage staying near SCA)
        deviation = np.linalg.norm(action)
        reward -= 0.01 * deviation

        # Update state
        old_pos = self.uav_pos.copy()
        self.uav_pos = new_pos

        # Check termination
        done = self.step_count >= self.max_steps

        # Info dict
        info = {
            'throughput': new_throughput,
            'throughput_mbps': new_throughput / 1e6,
            'sca_throughput': self.sca_throughput,
            'sca_throughput_mbps': self.sca_throughput / 1e6,
            'improvement': improvement,
            'improvement_pct': 100 * improvement / self.sca_throughput if self.sca_throughput > 0 else 0,
            'floor_activated': floor_activated,
            'position': new_pos.copy(),
            'sca_position': self.sca_pos.copy(),
            'deviation': np.linalg.norm(new_pos - self.sca_pos)
        }

        return self._get_state(), reward, done, info

    def _generate_random_users(self) -> np.ndarray:
        """Generate random user positions on ground."""
        positions = np.zeros((self.num_users, 3))
        positions[:, 0] = np.random.uniform(0, self.params.area_size, self.num_users)
        positions[:, 1] = np.random.uniform(0, self.params.area_size, self.num_users)
        positions[:, 2] = 0  # Ground level
        return positions

    def _project_to_feasible(self, pos: np.ndarray) -> np.ndarray:
        """Project position to feasible region."""
        pos = pos.copy()
        pos[0] = np.clip(pos[0], 0, self.params.area_size)
        pos[1] = np.clip(pos[1], 0, self.params.area_size)
        pos[2] = np.clip(pos[2], self.params.h_min, self.params.h_max)
        return pos

    def _compute_path_loss(self, d: float) -> float:
        """
        Compute path loss in linear scale.

        PL(dB) = 20*log10(d) + 20*log10(fc) + 20*log10(4*pi/c)
        """
        if d < 1e-6:
            d = 1e-6  # Avoid log(0)

        PL_dB = (20 * np.log10(d) +
                 20 * np.log10(self.params.fc) +
                 20 * np.log10(4 * np.pi / self.params.c))

        return 10 ** (PL_dB / 10)

    def _compute_snr(self, P_tx: float, d: float, G_tx: float, G_rx: float) -> float:
        """Compute SNR for a link."""
        PL = self._compute_path_loss(d)
        return (P_tx * G_tx * G_rx) / (self.params.N0 * self.params.B * PL)

    def _compute_throughput(self, uav_pos: np.ndarray) -> float:
        """
        Compute total network throughput for given UAV position.

        Uses decode-and-forward relay: rate limited by weaker hop.
        """
        # Backhaul link: BS -> UAV
        d_bu = np.linalg.norm(uav_pos - self.bs_pos)
        snr_bu = self._compute_snr(
            self.params.P_bs, d_bu,
            self.params.G_bs, self.params.G_uav
        )

        # Sum rate across all users
        total_rate = 0.0
        for k in range(self.num_users):
            # Access link: UAV -> User k
            d_k = np.linalg.norm(uav_pos - self.user_positions[k])
            snr_k = self._compute_snr(
                self.params.P_uav, d_k,
                self.params.G_uav, self.params.G_user
            )

            # DF relay: rate limited by weaker hop
            effective_snr = min(snr_bu, snr_k)
            rate_k = self.params.B * np.log2(1 + effective_snr)
            total_rate += rate_k

        return total_rate

    def _compute_throughput_gradient(self, pos: np.ndarray, eps: float = 0.1) -> np.ndarray:
        """Compute numerical gradient of throughput at position."""
        grad = np.zeros(3)
        Phi_0 = self._compute_throughput(pos)

        for dim in range(3):
            pos_plus = pos.copy()
            pos_plus[dim] += eps
            grad[dim] = (self._compute_throughput(pos_plus) - Phi_0) / eps

        return grad

    def _run_sca(self, num_iters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Successive Convex Approximation to find baseline position.

        Returns:
            (optimal_position, gradient_direction)
        """
        # Initialize at user centroid
        centroid = np.mean(self.user_positions, axis=0)
        pos = np.array([centroid[0], centroid[1],
                       (self.params.h_min + self.params.h_max) / 2])

        step_size = 1.0

        for _ in range(num_iters):
            # Compute gradient
            grad = self._compute_throughput_gradient(pos)
            grad_norm = np.linalg.norm(grad)

            if grad_norm > 1e-8:
                # Gradient ascent step
                pos = pos + step_size * grad / grad_norm

                # Project to feasible region
                pos = self._project_to_feasible(pos)

            # Decay step size
            step_size *= 0.95

        # Final gradient direction (normalized)
        final_grad = self._compute_throughput_gradient(pos)
        grad_norm = np.linalg.norm(final_grad)
        if grad_norm > 1e-8:
            final_grad = final_grad / grad_norm
        else:
            final_grad = np.zeros(3)

        return pos, final_grad

    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        # Current SNRs
        d_bu = np.linalg.norm(self.uav_pos - self.bs_pos)
        snr_bu = self._compute_snr(
            self.params.P_bs, d_bu,
            self.params.G_bs, self.params.G_uav
        )

        snr_users = np.zeros(self.num_users)
        for k in range(self.num_users):
            d_k = np.linalg.norm(self.uav_pos - self.user_positions[k])
            snr_users[k] = self._compute_snr(
                self.params.P_uav, d_k,
                self.params.G_uav, self.params.G_user
            )

        current_throughput = self._compute_throughput(self.uav_pos)

        # Normalize components for neural network input
        state = np.concatenate([
            self.uav_pos / self.params.area_size,           # [0, 1] range
            self.sca_pos / self.params.area_size,           # [0, 1] range
            self.sca_grad,                                   # Already normalized
            [np.log10(snr_bu + 1) / 10],                    # Log scale, ~[0, 1]
            np.log10(snr_users + 1) / 10,                   # Log scale
            [current_throughput / 1e9],                      # Gbps scale
            [self.sca_throughput / 1e9]                      # Gbps scale
        ])

        return state.astype(np.float32)

    def get_sca_action(self) -> np.ndarray:
        """Get the SCA gradient direction as action (for guidance)."""
        return self.sca_grad.copy()


class ScenarioGenerator:
    """Generate diverse scenarios for training."""

    def __init__(self, num_users: int = 5, area_size: float = 100.0, seed: int = 42):
        self.num_users = num_users
        self.area_size = area_size
        self.rng = np.random.default_rng(seed)

    def generate(self, num_scenarios: int) -> List[np.ndarray]:
        """Generate multiple random scenarios."""
        scenarios = []
        for _ in range(num_scenarios):
            positions = np.zeros((self.num_users, 3))
            positions[:, 0] = self.rng.uniform(0, self.area_size, self.num_users)
            positions[:, 1] = self.rng.uniform(0, self.area_size, self.num_users)
            scenarios.append(positions)
        return scenarios

    def generate_clustered(self, num_scenarios: int, num_clusters: int = 2) -> List[np.ndarray]:
        """Generate scenarios with clustered users."""
        scenarios = []
        for _ in range(num_scenarios):
            positions = np.zeros((self.num_users, 3))

            # Generate cluster centers
            centers = self.rng.uniform(20, 80, (num_clusters, 2))

            # Assign users to clusters
            for k in range(self.num_users):
                cluster = k % num_clusters
                positions[k, 0] = centers[cluster, 0] + self.rng.normal(0, 10)
                positions[k, 1] = centers[cluster, 1] + self.rng.normal(0, 10)
                positions[k, :2] = np.clip(positions[k, :2], 0, self.area_size)

            scenarios.append(positions)
        return scenarios


if __name__ == "__main__":
    # Test environment
    print("Testing UAV Relay Environment...")

    env = UAVRelayEnv(num_users=5, seed=42)
    state = env.reset()

    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Initial state shape: {state.shape}")
    print(f"SCA throughput: {env.sca_throughput / 1e6:.2f} Mbps")

    # Test random actions
    total_reward = 0
    for step in range(10):
        action = np.random.randn(3) * 0.1
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {step+1}: reward={reward:.4f}, throughput={info['throughput_mbps']:.2f} Mbps, "
              f"floor={info['floor_activated']}")

    print(f"\nTotal reward: {total_reward:.4f}")
    print("Environment test passed!")
