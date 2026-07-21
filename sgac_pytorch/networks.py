"""
Neural Network Architectures for SGAC Algorithm.
Implements Actor and Twin Critic networks following TD3 design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def weight_init(m: nn.Module):
    """Initialize network weights using orthogonal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Actor(nn.Module):
    """
    Actor network for SGAC.

    Outputs continuous actions (position corrections) given state.
    Uses tanh activation to bound actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_action: float = 1.0
    ):
        super().__init__()

        self.max_action = max_action

        # Network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self.apply(weight_init)

        # Small initialization for output layer (helps early training)
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_out.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Action tensor [batch_size, action_dim] in [-max_action, max_action]
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        action = self.max_action * torch.tanh(self.fc_out(x))
        return action


class Critic(nn.Module):
    """
    Critic network for SGAC.

    Estimates Q-value for state-action pairs.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Network layers
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self.apply(weight_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]

        Returns:
            Q-value tensor [batch_size, 1]
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        q_value = self.fc_out(x)
        return q_value


class TwinCritic(nn.Module):
    """
    Twin Critic networks for TD3-style training.

    Uses two independent critics and takes minimum to reduce overestimation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critics.

        Returns:
            (Q1, Q2) tuple of Q-value tensors
        """
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value from first critic only (for actor update)."""
        return self.critic1(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get minimum Q-value (for target computation)."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class SGACPolicy(nn.Module):
    """
    SGAC Hybrid Policy combining SCA guidance with learned corrections.

    policy(s) = alpha * sca_direction + beta * actor(s)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        alpha_sca: float = 0.7,
        beta_nn: float = 0.3,
        max_action: float = 1.0
    ):
        super().__init__()

        self.alpha_sca = alpha_sca
        self.beta_nn = beta_nn
        self.max_action = max_action

        # Actor network for learned corrections
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action)

        # Index positions for SCA direction in state vector
        # State format: [uav_pos(3), sca_pos(3), sca_grad(3), ...]
        self.sca_grad_start = 6
        self.sca_grad_end = 9

    def forward(
        self,
        state: torch.Tensor,
        sca_direction: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute hybrid action.

        Args:
            state: State tensor [batch_size, state_dim]
            sca_direction: Optional SCA gradient direction [batch_size, action_dim]
                          If None, extracts from state.

        Returns:
            Hybrid action tensor [batch_size, action_dim]
        """
        # Get learned correction from actor
        learned_correction = self.actor(state)

        # Extract SCA direction from state if not provided
        if sca_direction is None:
            sca_direction = state[:, self.sca_grad_start:self.sca_grad_end]

        # Hybrid action
        action = self.alpha_sca * sca_direction + self.beta_nn * learned_correction

        # Clip to valid range
        action = torch.clamp(action, -self.max_action, self.max_action)

        return action

    def get_learned_correction(self, state: torch.Tensor) -> torch.Tensor:
        """Get only the learned correction component."""
        return self.actor(state)

    def set_mixing_weights(self, alpha_sca: float, beta_nn: float):
        """Update mixing weights (for curriculum learning)."""
        self.alpha_sca = alpha_sca
        self.beta_nn = beta_nn


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.

    Stores transitions and samples random mini-batches.
    """

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

        # Pre-allocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        # Store SCA directions separately for hybrid policy
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
        """Add transition to buffer."""
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
        """Sample random batch of transitions."""
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


if __name__ == "__main__":
    # Test networks
    print("Testing SGAC Networks...")

    state_dim = 17  # Example dimension
    action_dim = 3
    batch_size = 32

    # Test Actor
    actor = Actor(state_dim, action_dim)
    state = torch.randn(batch_size, state_dim)
    action = actor(state)
    print(f"Actor output shape: {action.shape}")
    print(f"Actor output range: [{action.min().item():.3f}, {action.max().item():.3f}]")

    # Test Twin Critic
    critic = TwinCritic(state_dim, action_dim)
    q1, q2 = critic(state, action)
    print(f"Critic Q1 shape: {q1.shape}")
    print(f"Critic Q2 shape: {q2.shape}")

    # Test SGAC Policy
    policy = SGACPolicy(state_dim, action_dim)
    hybrid_action = policy(state)
    print(f"SGAC Policy output shape: {hybrid_action.shape}")

    # Test Replay Buffer
    buffer = ReplayBuffer(state_dim, action_dim, capacity=1000)
    for _ in range(100):
        buffer.add(
            np.random.randn(state_dim),
            np.random.randn(action_dim),
            np.random.randn(),
            np.random.randn(state_dim),
            False
        )

    batch = buffer.sample(32)
    print(f"Buffer size: {len(buffer)}")
    print(f"Batch states shape: {batch[0].shape}")

    print("\nAll network tests passed!")
