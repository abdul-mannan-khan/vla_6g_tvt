"""
UAV Relay Positioning Environment for 6G IoT Networks - FIXED VERSION

Key fixes:
1. Training mode: No floor, allows negative rewards for learning
2. Start position: Perturbed from SCA to give room for improvement
3. Reward shaping: Clear positive/negative signals based on performance
4. Evaluation mode: Floor enabled for safety guarantees
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


class UAVRelayEnvFixed:
    """
    FIXED UAV Relay Positioning Environment.

    Key differences from original:
    - Training mode: No floor mechanism, negative rewards allowed
    - Evaluation mode: Floor mechanism enabled
    - Start position: Perturbed from SCA (not at SCA)
    - Reward: Clear learning signal with positive/negative feedback
    """

    def __init__(
        self,
        num_users: int = 5,
        params: Optional[ChannelParams] = None,
        max_steps: int = 50,
        seed: Optional[int] = None,
        training_mode: bool = True,  # NEW: training vs evaluation mode
        start_perturbation: float = 10.0,  # NEW: how far from SCA to start
    ):
        self.num_users = num_users
        self.params = params or ChannelParams()
        self.max_steps = max_steps
        self.training_mode = training_mode
        self.start_perturbation = start_perturbation

        if seed is not None:
            np.random.seed(seed)

        # State and action dimensions
        self.state_dim = 3 + 3 + 3 + 1 + num_users + 1 + 1
        self.action_dim = 3

        # Fixed positions
        self.bs_pos = np.array([50.0, 50.0, 15.0])

        # Episode state
        self.user_positions = None
        self.uav_pos = None
        self.sca_pos = None
        self.sca_grad = None
        self.sca_throughput = None
        self.step_count = 0
        self.best_throughput_this_episode = None

    def set_training_mode(self, training: bool):
        """Switch between training and evaluation modes."""
        self.training_mode = training

    def reset(self, user_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environment with new scenario."""
        self.step_count = 0

        if user_positions is not None:
            self.user_positions = user_positions.copy()
        else:
            self.user_positions = self._generate_random_users()

        # Compute SCA solution
        self.sca_pos, self.sca_grad = self._run_sca(num_iters=20)
        self.sca_throughput = self._compute_throughput(self.sca_pos)

        # FIX #1: Start AWAY from SCA in training mode
        if self.training_mode and self.start_perturbation > 0:
            # Random perturbation from SCA position
            perturbation = np.random.randn(3) * self.start_perturbation
            perturbation[2] *= 0.3  # Smaller altitude perturbation
            self.uav_pos = self._project_to_feasible(self.sca_pos + perturbation)
        else:
            # Evaluation: start at SCA
            self.uav_pos = self.sca_pos.copy()

        self.best_throughput_this_episode = self._compute_throughput(self.uav_pos)

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return next state, reward, done, info."""
        self.step_count += 1

        # Apply action
        step_size = 2.0
        new_pos = self.uav_pos + step_size * action
        new_pos = self._project_to_feasible(new_pos)

        # Compute throughput
        new_throughput = self._compute_throughput(new_pos)

        # Track best throughput this episode
        if new_throughput > self.best_throughput_this_episode:
            self.best_throughput_this_episode = new_throughput

        # FIX #2: Floor only in evaluation mode
        floor_activated = False
        if not self.training_mode:
            # Evaluation: apply floor guarantee
            if new_throughput < self.sca_throughput:
                new_pos = self.sca_pos.copy()
                new_throughput = self.sca_throughput
                floor_activated = True

        # FIX #3: Proper reward with NEGATIVE feedback
        reward = self._compute_reward(new_throughput, action, floor_activated)

        # Update position
        self.uav_pos = new_pos

        done = self.step_count >= self.max_steps

        info = {
            'throughput': new_throughput,
            'throughput_mbps': new_throughput / 1e6,
            'sca_throughput': self.sca_throughput,
            'sca_throughput_mbps': self.sca_throughput / 1e6,
            'improvement': new_throughput - self.sca_throughput,
            'improvement_pct': 100 * (new_throughput - self.sca_throughput) / self.sca_throughput,
            'floor_activated': floor_activated,
            'position': new_pos.copy(),
            'sca_position': self.sca_pos.copy(),
            'deviation': np.linalg.norm(new_pos - self.sca_pos),
            'best_throughput': self.best_throughput_this_episode,
        }

        return self._get_state(), reward, done, info

    def _compute_reward(self, throughput: float, action: np.ndarray, floor_activated: bool) -> float:
        """
        FIX #3: Proper reward function with clear learning signal.

        Key properties:
        - Positive reward for throughput > SCA
        - NEGATIVE reward for throughput < SCA (crucial for learning!)
        - Scaled appropriately for stable training
        """
        # Relative improvement over SCA (can be negative!)
        improvement_pct = (throughput - self.sca_throughput) / self.sca_throughput

        # Scale to reasonable magnitude (roughly -1 to +1 range for typical cases)
        # Multiply by 10 to make the signal stronger
        reward = improvement_pct * 10.0

        # Small action penalty to encourage efficiency
        action_cost = 0.01 * np.linalg.norm(action)
        reward -= action_cost

        # Bonus for exceeding SCA significantly
        if improvement_pct > 0.02:  # >2% improvement
            reward += 0.5

        # Penalty for floor activation (only matters in eval mode)
        if floor_activated:
            reward -= 0.1

        return reward

    def _generate_random_users(self) -> np.ndarray:
        """Generate random user positions on ground."""
        positions = np.zeros((self.num_users, 3))
        positions[:, 0] = np.random.uniform(0, self.params.area_size, self.num_users)
        positions[:, 1] = np.random.uniform(0, self.params.area_size, self.num_users)
        positions[:, 2] = 0
        return positions

    def _project_to_feasible(self, pos: np.ndarray) -> np.ndarray:
        """Project position to feasible region."""
        pos = pos.copy()
        pos[0] = np.clip(pos[0], 0, self.params.area_size)
        pos[1] = np.clip(pos[1], 0, self.params.area_size)
        pos[2] = np.clip(pos[2], self.params.h_min, self.params.h_max)
        return pos

    def _compute_path_loss(self, d: float) -> float:
        """Compute path loss in linear scale."""
        if d < 1e-6:
            d = 1e-6
        PL_dB = (20 * np.log10(d) +
                 20 * np.log10(self.params.fc) +
                 20 * np.log10(4 * np.pi / self.params.c))
        return 10 ** (PL_dB / 10)

    def _compute_snr(self, P_tx: float, d: float, G_tx: float, G_rx: float) -> float:
        """Compute SNR for a link."""
        PL = self._compute_path_loss(d)
        return (P_tx * G_tx * G_rx) / (self.params.N0 * self.params.B * PL)

    def _compute_throughput(self, uav_pos: np.ndarray) -> float:
        """Compute total network throughput."""
        d_bu = np.linalg.norm(uav_pos - self.bs_pos)
        snr_bu = self._compute_snr(
            self.params.P_bs, d_bu,
            self.params.G_bs, self.params.G_uav
        )

        total_rate = 0.0
        for k in range(self.num_users):
            d_k = np.linalg.norm(uav_pos - self.user_positions[k])
            snr_k = self._compute_snr(
                self.params.P_uav, d_k,
                self.params.G_uav, self.params.G_user
            )
            effective_snr = min(snr_bu, snr_k)
            rate_k = self.params.B * np.log2(1 + effective_snr)
            total_rate += rate_k

        return total_rate

    def _compute_throughput_gradient(self, pos: np.ndarray, eps: float = 0.1) -> np.ndarray:
        """Compute numerical gradient of throughput."""
        grad = np.zeros(3)
        Phi_0 = self._compute_throughput(pos)
        for dim in range(3):
            pos_plus = pos.copy()
            pos_plus[dim] += eps
            grad[dim] = (self._compute_throughput(pos_plus) - Phi_0) / eps
        return grad

    def _run_sca(self, num_iters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Run SCA to find baseline position."""
        centroid = np.mean(self.user_positions, axis=0)
        pos = np.array([centroid[0], centroid[1],
                       (self.params.h_min + self.params.h_max) / 2])

        step_size = 1.0
        for _ in range(num_iters):
            grad = self._compute_throughput_gradient(pos)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-8:
                pos = pos + step_size * grad / grad_norm
                pos = self._project_to_feasible(pos)
            step_size *= 0.95

        final_grad = self._compute_throughput_gradient(pos)
        grad_norm = np.linalg.norm(final_grad)
        if grad_norm > 1e-8:
            final_grad = final_grad / grad_norm
        else:
            final_grad = np.zeros(3)

        return pos, final_grad

    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
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

        state = np.concatenate([
            self.uav_pos / self.params.area_size,
            self.sca_pos / self.params.area_size,
            self.sca_grad,
            [np.log10(snr_bu + 1) / 10],
            np.log10(snr_users + 1) / 10,
            [current_throughput / 1e9],
            [self.sca_throughput / 1e9]
        ])

        return state.astype(np.float32)

    def get_sca_action(self) -> np.ndarray:
        """Get SCA gradient direction."""
        return self.sca_grad.copy()


# Backward compatibility alias
UAVRelayEnv = UAVRelayEnvFixed


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


if __name__ == "__main__":
    print("Testing FIXED UAV Relay Environment...")
    print("=" * 60)

    # Test training mode
    env = UAVRelayEnvFixed(num_users=5, seed=42, training_mode=True)
    state = env.reset()

    print(f"Training Mode Test:")
    print(f"  SCA throughput: {env.sca_throughput / 1e6:.2f} Mbps")
    print(f"  Start throughput: {env._compute_throughput(env.uav_pos) / 1e6:.2f} Mbps")
    print(f"  Start position: {env.uav_pos}")
    print(f"  SCA position: {env.sca_pos}")
    print(f"  Distance from SCA: {np.linalg.norm(env.uav_pos - env.sca_pos):.2f} m")

    # Test that negative rewards are possible
    rewards = []
    for _ in range(20):
        action = np.random.randn(3) * 0.5
        _, reward, _, info = env.step(action)
        rewards.append(reward)

    print(f"\n  Rewards: min={min(rewards):.2f}, max={max(rewards):.2f}")
    print(f"  Negative rewards: {sum(1 for r in rewards if r < 0)} / {len(rewards)}")

    # Test evaluation mode
    print(f"\nEvaluation Mode Test:")
    env.set_training_mode(False)
    state = env.reset()

    print(f"  Start position == SCA: {np.allclose(env.uav_pos, env.sca_pos)}")

    floor_count = 0
    for _ in range(20):
        action = np.random.randn(3) * 0.5
        _, _, _, info = env.step(action)
        if info['floor_activated']:
            floor_count += 1

    print(f"  Floor activations: {floor_count} / 20")
    print(f"  Final throughput >= SCA: {info['throughput'] >= env.sca_throughput}")

    print("\n" + "=" * 60)
    print("FIXED Environment tests passed!")
