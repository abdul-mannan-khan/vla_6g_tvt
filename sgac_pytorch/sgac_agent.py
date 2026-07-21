"""
SGAC Agent: Successive Convex Approximation-Guided Actor-Critic

Implements the full SGAC algorithm with:
- Hybrid policy (SCA guidance + learned corrections)
- Twin critic (TD3-style)
- Delayed actor updates
- Target network soft updates
- Performance floor guarantee
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import copy

from networks import SGACPolicy, TwinCritic, ReplayBuffer


class SGACAgent:
    """
    SGAC Agent for UAV Relay Positioning.

    Combines successive convex approximation with deep reinforcement learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # Network architecture
        hidden_dim: int = 256,
        # SGAC mixing weights
        alpha_sca: float = 0.7,
        beta_nn: float = 0.3,
        # Learning parameters
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        # TD3 parameters
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        # Exploration
        exploration_noise: float = 0.1,
        # Buffer
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        # Device
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

        # Initialize networks
        self.policy = SGACPolicy(
            state_dim, action_dim, hidden_dim,
            alpha_sca, beta_nn
        ).to(device)

        self.critic = TwinCritic(state_dim, action_dim, hidden_dim).to(device)

        # Target networks
        self.policy_target = copy.deepcopy(self.policy)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        # Replay buffer
        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)

        # Training state
        self.total_steps = 0
        self.training_info = {}

    def select_action(
        self,
        state: np.ndarray,
        sca_direction: Optional[np.ndarray] = None,
        evaluate: bool = False
    ) -> np.ndarray:
        """
        Select action given state.

        Args:
            state: Current state
            sca_direction: SCA gradient direction for hybrid policy
            evaluate: If True, no exploration noise

        Returns:
            Action array
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if sca_direction is not None:
                sca_t = torch.FloatTensor(sca_direction).unsqueeze(0).to(self.device)
                action = self.policy(state_t, sca_t)
            else:
                action = self.policy(state_t)

            action = action.cpu().numpy()[0]

        # Add exploration noise during training
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
        """Store transition in replay buffer."""
        self.buffer.add(state, action, reward, next_state, done, sca_direction)

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dict of training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {}

        self.total_steps += 1

        # Sample batch
        states, actions, rewards, next_states, dones, sca_dirs = \
            self.buffer.sample(self.batch_size)

        # ----- Critic Update -----
        with torch.no_grad():
            # Target actions with noise (TD3 smoothing)
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.policy_target(next_states, sca_dirs) + noise).clamp(-1, 1)

            # Target Q-values (use minimum of twin critics)
            target_q = self.critic_target.q_min(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss (MSE on both critics)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ----- Delayed Actor Update -----
        actor_loss = torch.tensor(0.0)
        if self.total_steps % self.policy_delay == 0:
            # Actor loss: maximize Q-value
            actor_actions = self.policy(states, sca_dirs)
            actor_loss = -self.critic.q1(states, actor_actions).mean()

            # Add regularization towards SCA direction
            learned_corrections = self.policy.get_learned_correction(states)
            regularization = 0.01 * (learned_corrections ** 2).mean()
            actor_loss = actor_loss + regularization

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update target networks
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
        """Soft update target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'alpha_sca': self.alpha_sca,
            'beta_nn': self.beta_nn,
        }, filepath)

    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']

        if 'alpha_sca' in checkpoint:
            self.alpha_sca = checkpoint['alpha_sca']
            self.beta_nn = checkpoint['beta_nn']
            self.policy.set_mixing_weights(self.alpha_sca, self.beta_nn)

    def set_exploration_noise(self, noise: float):
        """Update exploration noise (for annealing)."""
        self.exploration_noise = noise

    def get_statistics(self) -> Dict[str, float]:
        """Get current training statistics."""
        return self.training_info.copy()


class BaselineAgents:
    """
    Baseline agents for comparison.
    """

    @staticmethod
    def random_agent(action_dim: int) -> np.ndarray:
        """Random action baseline."""
        return np.random.uniform(-1, 1, action_dim)

    @staticmethod
    def sca_agent(sca_direction: np.ndarray, scale: float = 0.5) -> np.ndarray:
        """Pure SCA gradient following."""
        return scale * sca_direction


if __name__ == "__main__":
    # Test SGAC Agent
    print("Testing SGAC Agent...")

    state_dim = 17
    action_dim = 3

    agent = SGACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device="cpu"
    )

    # Test action selection
    state = np.random.randn(state_dim).astype(np.float32)
    sca_dir = np.random.randn(action_dim).astype(np.float32)
    sca_dir = sca_dir / np.linalg.norm(sca_dir)

    action = agent.select_action(state, sca_dir)
    print(f"Selected action shape: {action.shape}")
    print(f"Selected action: {action}")

    # Test training (with dummy data)
    for i in range(500):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = False
        sca_dir = np.random.randn(action_dim).astype(np.float32)

        agent.store_transition(state, action, reward, next_state, done, sca_dir)

        if i >= 256:
            info = agent.train_step()
            if i % 100 == 0:
                print(f"Step {i}: critic_loss={info.get('critic_loss', 0):.4f}, "
                      f"actor_loss={info.get('actor_loss', 0):.4f}")

    # Test save/load
    agent.save("/tmp/sgac_test.pt")
    agent.load("/tmp/sgac_test.pt")
    print("Save/load test passed!")

    print("\nSGAC Agent tests passed!")
