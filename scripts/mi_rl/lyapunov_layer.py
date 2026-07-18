#!/usr/bin/env python3
"""
Lyapunov Safety Layer for Constrained RL.

Ensures that actions taken by the RL policy satisfy stability constraints.
The key idea is to project actions onto the safe set defined by:

    V(s_{t+1}) - V(s_t) <= -gamma * V(s_t)

This guarantees exponential stability of the system.

Reference: Chow et al., "Lyapunov-based Safe Policy Optimization" (2019)
"""

import numpy as np
from typing import Tuple, Optional
import torch
import torch.nn as nn


class LyapunovFunction(nn.Module):
    """
    Learnable Lyapunov function V(s) that measures "badness" of state.

    For UAV positioning, we define:
    V(s) = ||p - p_target||^2 + alpha * variance(user_rates)

    This encourages:
    1. Moving toward target position
    2. Fair distribution of rates across users
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure V(s) >= 0
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class AnalyticalLyapunov:
    """
    Analytical Lyapunov function based on domain knowledge.

    V(p) = w1 * ||p - p_centroid||^2 + w2 * (max_distance - optimal_distance)^2

    This is differentiable and doesn't require learning.
    """

    def __init__(self, w1: float = 1.0, w2: float = 0.5):
        self.w1 = w1
        self.w2 = w2

    def __call__(self, p: np.ndarray, scenario) -> float:
        """Compute Lyapunov value at position p."""
        # Distance to user centroid
        user_centroid = np.mean(scenario.user_positions, axis=0)[:2]
        d_centroid = np.linalg.norm(p[:2] - user_centroid)

        # Height deviation from optimal
        optimal_z = np.clip(0.5 * d_centroid, 10, 40)
        z_deviation = abs(p[2] - optimal_z)

        # Variance in user distances (promotes fairness)
        user_distances = [np.linalg.norm(p - u) for u in scenario.user_positions]
        distance_variance = np.var(user_distances) if len(user_distances) > 1 else 0

        return self.w1 * (d_centroid ** 2 + z_deviation ** 2) + self.w2 * distance_variance

    def gradient(self, p: np.ndarray, scenario, eps: float = 0.01) -> np.ndarray:
        """Compute gradient of Lyapunov function."""
        grad = np.zeros(3)
        v0 = self(p, scenario)

        for i in range(3):
            p_plus = p.copy()
            p_plus[i] += eps
            grad[i] = (self(p_plus, scenario) - v0) / eps

        return grad


class LyapunovSafetyLayer:
    """
    Projects actions to satisfy Lyapunov decrease condition.

    Given action a from policy, finds closest safe action a' such that:
    V(f(s, a')) - V(s) <= -gamma * V(s)

    where f(s, a) is the transition dynamics.
    """

    def __init__(self, gamma: float = 0.1, max_projection_steps: int = 10):
        """
        Args:
            gamma: Exponential decay rate for Lyapunov function
            max_projection_steps: Maximum iterations for projection
        """
        self.gamma = gamma
        self.max_projection_steps = max_projection_steps
        self.lyapunov = AnalyticalLyapunov()

    def project_action(self, current_pos: np.ndarray,
                       proposed_action: np.ndarray,
                       scenario) -> Tuple[np.ndarray, bool]:
        """
        Project proposed action to safe set.

        Args:
            current_pos: Current UAV position
            proposed_action: Proposed position change (delta)
            scenario: Current scenario

        Returns:
            safe_action: Projected action satisfying Lyapunov constraint
            is_safe: Whether original action was already safe
        """
        # Current Lyapunov value
        v_current = self.lyapunov(current_pos, scenario)

        # Proposed next position
        proposed_pos = self._apply_action(current_pos, proposed_action)
        v_proposed = self.lyapunov(proposed_pos, scenario)

        # Check if already safe
        lyapunov_decrease = v_proposed - v_current
        threshold = -self.gamma * v_current

        if lyapunov_decrease <= threshold:
            return proposed_action, True

        # Not safe - need to project
        # Use gradient projection to find closest safe action
        safe_action = self._gradient_projection(
            current_pos, proposed_action, scenario, v_current
        )

        return safe_action, False

    def _apply_action(self, pos: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply action to get next position with constraints."""
        next_pos = pos + action
        # Clip to feasible region
        next_pos[0] = np.clip(next_pos[0], 0, 100)
        next_pos[1] = np.clip(next_pos[1], 0, 100)
        next_pos[2] = np.clip(next_pos[2], 10, 40)
        return next_pos

    def _gradient_projection(self, current_pos: np.ndarray,
                             proposed_action: np.ndarray,
                             scenario,
                             v_current: float) -> np.ndarray:
        """
        Project action using gradient descent on constraint violation.

        min ||a - a_proposed||^2
        s.t. V(f(s, a)) - V(s) <= -gamma * V(s)
        """
        action = proposed_action.copy()
        threshold = -self.gamma * v_current

        for _ in range(self.max_projection_steps):
            next_pos = self._apply_action(current_pos, action)
            v_next = self.lyapunov(next_pos, scenario)
            violation = v_next - v_current - threshold

            if violation <= 0:
                break

            # Gradient of violation w.r.t. action
            grad_v = self.lyapunov.gradient(next_pos, scenario)

            # Project action in direction of decreasing Lyapunov
            step_size = 0.1 * violation / (np.linalg.norm(grad_v) ** 2 + 1e-8)
            action = action - step_size * grad_v

            # Keep action magnitude reasonable
            action_norm = np.linalg.norm(action)
            if action_norm > 10:
                action = action * 10 / action_norm

        return action

    def check_safety(self, current_pos: np.ndarray,
                     action: np.ndarray,
                     scenario) -> Tuple[bool, float]:
        """
        Check if action satisfies Lyapunov constraint.

        Returns:
            is_safe: Whether action is safe
            margin: How much margin to constraint (negative = violation)
        """
        v_current = self.lyapunov(current_pos, scenario)
        next_pos = self._apply_action(current_pos, action)
        v_next = self.lyapunov(next_pos, scenario)

        threshold = -self.gamma * v_current
        margin = threshold - (v_next - v_current)

        return margin >= 0, margin


class LyapunovConstrainedPolicy(nn.Module):
    """
    Neural network policy with built-in Lyapunov safety layer.

    The policy outputs an unconstrained action, which is then
    projected to the safe set by the Lyapunov layer.
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_dim: int = 128, gamma: float = 0.1):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1], scaled later
        )

        self.action_scale = 5.0  # Maximum action magnitude
        self.safety_layer = LyapunovSafetyLayer(gamma=gamma)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get unconstrained action (for training)."""
        return self.policy_net(state) * self.action_scale

    def get_safe_action(self, state: torch.Tensor,
                        current_pos: np.ndarray,
                        scenario) -> Tuple[np.ndarray, bool]:
        """
        Get safe action after Lyapunov projection.

        Args:
            state: State tensor
            current_pos: Current UAV position
            scenario: Scenario object

        Returns:
            safe_action: Action satisfying Lyapunov constraint
            was_projected: Whether projection was needed
        """
        with torch.no_grad():
            unconstrained_action = self.forward(state).numpy()

        safe_action, is_safe = self.safety_layer.project_action(
            current_pos, unconstrained_action, scenario
        )

        return safe_action, not is_safe


if __name__ == "__main__":
    """Test Lyapunov safety layer."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval_common import generate_scenarios

    scenarios = generate_scenarios(num_scenarios=5)

    print("Testing Lyapunov Safety Layer")
    print("=" * 60)

    safety_layer = LyapunovSafetyLayer(gamma=0.1)

    for scenario in scenarios[:3]:
        p = scenario.initial_uav_position

        # Random proposed action
        proposed_action = np.random.randn(3) * 5

        print(f"Scenario {scenario.id}:")
        print(f"  Current pos: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
        print(f"  Proposed action: [{proposed_action[0]:.2f}, {proposed_action[1]:.2f}, {proposed_action[2]:.2f}]")

        # Check original safety
        is_safe_orig, margin_orig = safety_layer.check_safety(p, proposed_action, scenario)
        print(f"  Original safe: {is_safe_orig}, margin: {margin_orig:.3f}")

        # Project to safe
        safe_action, was_safe = safety_layer.project_action(p, proposed_action, scenario)
        print(f"  Safe action: [{safe_action[0]:.2f}, {safe_action[1]:.2f}, {safe_action[2]:.2f}]")

        # Verify safety
        is_safe_proj, margin_proj = safety_layer.check_safety(p, safe_action, scenario)
        print(f"  Projected safe: {is_safe_proj}, margin: {margin_proj:.3f}")

        # Lyapunov values
        lyap = safety_layer.lyapunov
        v_curr = lyap(p, scenario)
        v_next_orig = lyap(p + proposed_action, scenario)
        v_next_safe = lyap(p + safe_action, scenario)
        print(f"  V(current): {v_curr:.2f}")
        print(f"  V(proposed): {v_next_orig:.2f} (change: {v_next_orig - v_curr:.2f})")
        print(f"  V(safe): {v_next_safe:.2f} (change: {v_next_safe - v_curr:.2f})")
        print()
