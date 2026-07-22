"""
Deep Deterministic Policy Gradient (DDPG) Agent
Baseline implementation for comparison with SGAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from collections import deque
import random


class Actor(nn.Module):
    """DDPG Actor network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """DDPG Critic network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


class DDPGAgent:
    """DDPG Agent for UAV positioning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
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

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.actor_loss = 0
        self.critic_loss = 0

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        """Select action with exploration noise."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        # Add exploration noise
        if noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)

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
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
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
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
