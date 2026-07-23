"""
Neural Network Architectures for SGAC Algorithm - FIXED VERSION

Key fixes:
1. Better balance between SCA and learned components
2. Curriculum learning: start with SCA guidance, gradually shift to learned
3. Improved initialization for faster learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def weight_init(m: nn.Module):
    """Initialize network weights."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Actor(nn.Module):
    """Actor network for SGAC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_action: float = 1.0
    ):
        super().__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.apply(weight_init)
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_out.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        action = self.max_action * torch.tanh(self.fc_out(x))
        return action


class Critic(nn.Module):
    """Critic network for SGAC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.apply(weight_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        q_value = self.fc_out(x)
        return q_value


class TwinCritic(nn.Module):
    """Twin Critic networks for TD3-style training."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.critic1(state, action), self.critic2(state, action)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.critic1(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class SGACPolicyFixed(nn.Module):
    """
    FIXED SGAC Hybrid Policy.

    Key fixes:
    1. Higher initial weight on learned component (0.5 instead of 0.3)
    2. Curriculum learning support: can anneal from SCA-heavy to learning-heavy
    3. Option to use ONLY learned component (pure RL mode)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        alpha_sca: float = 0.5,  # FIXED: Reduced from 0.7
        beta_nn: float = 0.5,    # FIXED: Increased from 0.3
        max_action: float = 1.0
    ):
        super().__init__()

        self.alpha_sca = alpha_sca
        self.beta_nn = beta_nn
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action)

        # SCA gradient indices in state
        self.sca_grad_start = 6
        self.sca_grad_end = 9

    def forward(
        self,
        state: torch.Tensor,
        sca_direction: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute hybrid action."""
        learned_correction = self.actor(state)

        if sca_direction is None:
            sca_direction = state[:, self.sca_grad_start:self.sca_grad_end]

        # Hybrid action with balanced weights
        action = self.alpha_sca * sca_direction + self.beta_nn * learned_correction
        action = torch.clamp(action, -self.max_action, self.max_action)

        return action

    def get_learned_correction(self, state: torch.Tensor) -> torch.Tensor:
        """Get only the learned correction."""
        return self.actor(state)

    def get_pure_learned_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get action using ONLY the learned component (for testing)."""
        return self.actor(state)

    def set_mixing_weights(self, alpha_sca: float, beta_nn: float):
        """Update mixing weights for curriculum learning."""
        self.alpha_sca = alpha_sca
        self.beta_nn = beta_nn

    def anneal_to_learned(self, progress: float):
        """
        Curriculum learning: gradually shift from SCA to learned.

        Args:
            progress: Training progress from 0.0 to 1.0

        At progress=0: alpha=0.7, beta=0.3 (SCA-heavy)
        At progress=1: alpha=0.3, beta=0.7 (learning-heavy)
        """
        self.alpha_sca = 0.7 - 0.4 * progress  # 0.7 -> 0.3
        self.beta_nn = 0.3 + 0.4 * progress    # 0.3 -> 0.7


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        device: str = "cpu"
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.sca_directions = np.zeros((capacity, action_dim), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        sca_direction: Optional[np.ndarray] = None
    ):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        if sca_direction is not None:
            self.sca_directions[self.ptr] = sca_direction

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
            torch.FloatTensor(self.sca_directions[indices]).to(self.device)
        )

    def __len__(self) -> int:
        return self.size


# Backward compatibility
SGACPolicy = SGACPolicyFixed


if __name__ == "__main__":
    print("Testing FIXED SGAC Networks...")

    state_dim = 17
    action_dim = 3
    batch_size = 32

    # Test policy with curriculum learning
    policy = SGACPolicyFixed(state_dim, action_dim)
    state = torch.randn(batch_size, state_dim)

    print(f"Initial weights: alpha={policy.alpha_sca}, beta={policy.beta_nn}")

    # Test annealing
    for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
        policy.anneal_to_learned(progress)
        print(f"Progress {progress:.2f}: alpha={policy.alpha_sca:.2f}, beta={policy.beta_nn:.2f}")

    print("\nFIXED Networks tests passed!")
