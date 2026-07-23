"""
SGAC Agent - FIXED VERSION

Key fixes:
1. Uses fixed policy with curriculum learning
2. Reduced regularization towards SCA (was preventing learning)
3. Better Q-value estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import copy

from networks_fixed import SGACPolicyFixed, TwinCritic, ReplayBuffer


class SGACAgentFixed:
    """FIXED SGAC Agent with proper learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        alpha_sca: float = 0.5,
        beta_nn: float = 0.5,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.device = device

        self.alpha_sca = alpha_sca
        self.beta_nn = beta_nn

        # FIXED: Use fixed policy
        self.policy = SGACPolicyFixed(
            state_dim, action_dim, hidden_dim,
            alpha_sca, beta_nn
        ).to(device)

        self.critic = TwinCritic(state_dim, action_dim, hidden_dim).to(device)

        self.policy_target = copy.deepcopy(self.policy)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)

        self.total_steps = 0
        self.training_info = {}

    def select_action(
        self,
        state: np.ndarray,
        sca_direction: Optional[np.ndarray] = None,
        evaluate: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if sca_direction is not None:
                sca_t = torch.FloatTensor(sca_direction).unsqueeze(0).to(self.device)
                action = self.policy(state_t, sca_t)
            else:
                action = self.policy(state_t)

            action = action.cpu().numpy()[0]

        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -1.0, 1.0)

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        sca_direction: Optional[np.ndarray] = None
    ):
        self.buffer.add(state, action, reward, next_state, done, sca_direction)

    def train_step(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}

        self.total_steps += 1

        states, actions, rewards, next_states, dones, sca_dirs = \
            self.buffer.sample(self.batch_size)

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.policy_target(next_states, sca_dirs) + noise).clamp(-1, 1)
            target_q = self.critic_target.q_min(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Delayed actor update
        actor_loss = torch.tensor(0.0)
        if self.total_steps % self.policy_delay == 0:
            actor_actions = self.policy(states, sca_dirs)
            actor_loss = -self.critic.q1(states, actor_actions).mean()

            # FIXED: Reduced regularization (was 0.01, now 0.001)
            # Too much regularization prevents learning!
            learned_corrections = self.policy.get_learned_correction(states)
            regularization = 0.001 * (learned_corrections ** 2).mean()
            actor_loss = actor_loss + regularization

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self._soft_update(self.policy, self.policy_target)
            self._soft_update(self.critic, self.critic_target)

        self.training_info = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
            'buffer_size': len(self.buffer)
        }

        return self.training_info

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filepath: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'alpha_sca': self.policy.alpha_sca,
            'beta_nn': self.policy.beta_nn,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']

        if 'alpha_sca' in checkpoint:
            self.policy.set_mixing_weights(
                checkpoint['alpha_sca'],
                checkpoint['beta_nn']
            )

    def set_exploration_noise(self, noise: float):
        self.exploration_noise = noise


if __name__ == "__main__":
    print("Testing FIXED SGAC Agent...")

    state_dim = 17
    action_dim = 3

    agent = SGACAgentFixed(
        state_dim=state_dim,
        action_dim=action_dim,
        device="cpu"
    )

    # Test with dummy data
    for i in range(500):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        # FIXED: Include negative rewards
        reward = np.random.randn() * 2  # Can be negative!
        next_state = np.random.randn(state_dim).astype(np.float32)
        sca_dir = np.random.randn(action_dim).astype(np.float32)

        agent.store_transition(state, action, reward, next_state, False, sca_dir)

        if i >= 256:
            info = agent.train_step()

    print("FIXED SGAC Agent tests passed!")
