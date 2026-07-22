"""
Proximal Policy Optimization (PPO) Agent
Baseline implementation for comparison with SGAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List


class ActorCritic(nn.Module):
    """PPO Actor-Critic network with shared features."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(features)
        return mean, std, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Get action and log probability."""
        mean, std, value = self.forward(state)

        if deterministic:
            action = torch.tanh(mean) * self.max_action
            return action, None, value

        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t) * self.max_action

        # Log probability with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update."""
        mean, std, values = self.forward(states)

        # Inverse tanh to get x_t
        actions_scaled = actions / self.max_action
        actions_scaled = actions_scaled.clamp(-0.999, 0.999)
        x_t = 0.5 * (torch.log(1 + actions_scaled) - torch.log(1 - actions_scaled))

        dist = Normal(mean, std)
        log_probs = dist.log_prob(x_t)
        log_probs -= torch.log(self.max_action * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_probs, values, entropy


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def compute_returns(self, next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE returns."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones + [False])

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values[:-1]
        return returns, advantages

    def get_batches(self, batch_size: int, returns: np.ndarray, advantages: np.ndarray):
        """Generate minibatches."""
        n_samples = len(self.states)
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                np.array([self.states[i] for i in batch_indices]),
                np.array([self.actions[i] for i in batch_indices]),
                np.array([self.log_probs[i] for i in batch_indices]),
                returns[batch_indices],
                advantages[batch_indices]
            )


class PPOAgent:
    """PPO Agent for UAV positioning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        max_action: float = 1.0,
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
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training stats
        self.actor_loss = 0
        self.critic_loss = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Select action."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_t, deterministic)

        action = action.cpu().numpy()[0]
        log_prob = log_prob.cpu().numpy()[0, 0] if log_prob is not None else 0
        value = value.cpu().numpy()[0, 0]

        return action, log_prob, value

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in rollout buffer."""
        self.buffer.push(state, action, reward, value, log_prob, done)

    def train_step(self) -> Tuple[float, float]:
        """Perform PPO update on collected rollout."""
        if len(self.buffer.states) == 0:
            return 0.0, 0.0

        # Get final value for GAE
        last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, last_value = self.policy.get_action(last_state, deterministic=True)
            last_value = last_value.cpu().numpy()[0, 0]

        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns(last_value, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        n_updates = 0

        # PPO epochs
        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size, returns, advantages):
                states, actions, old_log_probs, batch_returns, batch_advantages = batch

                states = torch.FloatTensor(states).to(self.device)
                actions = torch.FloatTensor(actions).to(self.device)
                old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1).to(self.device)
                batch_returns = torch.FloatTensor(batch_returns).unsqueeze(1).to(self.device)
                batch_advantages = torch.FloatTensor(batch_advantages).unsqueeze(1).to(self.device)

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(states, actions)

                # Policy loss (PPO clip)
                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_updates += 1

        # Clear buffer
        self.buffer.clear()

        self.actor_loss = total_actor_loss / max(n_updates, 1)
        self.critic_loss = total_critic_loss / max(n_updates, 1)

        return self.critic_loss, self.actor_loss

    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy': self.policy.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
