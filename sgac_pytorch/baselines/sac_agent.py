"""
Soft Actor-Critic (SAC) Agent
Baseline implementation for comparison with SGAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple
from collections import deque
import random

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    """SAC Gaussian Actor network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action

        # Log probability with tanh squashing correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for evaluation."""
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean) * self.max_action
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t) * self.max_action


class TwinCritic(nn.Module):
    """SAC Twin Critic network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)

        # Q2
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)

        q2 = F.relu(self.fc1_q2(x))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)

        return q1, q2


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """SAC Agent for UAV positioning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        max_action: float = 1.0,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy = auto_entropy

        # Networks
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.critic = TwinCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Entropy temperature
        if auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.actor_loss = 0
        self.critic_loss = 0
        self.alpha_loss = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor.get_action(state_t, deterministic).cpu().numpy()[0]

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Tuple[float, float]:
        """Perform one training step."""
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update entropy temperature
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            self.alpha_loss = alpha_loss.item()

        # Soft update target network
        self._soft_update(self.critic, self.critic_target)

        self.actor_loss = actor_loss.item()
        self.critic_loss = critic_loss.item()

        return self.critic_loss, self.actor_loss

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        """Save model."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy else None,
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        if self.auto_entropy and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
