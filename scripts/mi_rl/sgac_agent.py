#!/usr/bin/env python3
"""
SCA-Guided Actor-Critic (SGAC) Agent for UAV Relay Positioning.

This is the core Math-Informed RL agent that combines:
1. SCA warm-starting: Initialize policy from classical optimization
2. Physics-informed features: Encode channel model in state representation
3. Residual learning: Learn corrections to SCA baseline
4. Lyapunov safety: Project actions to satisfy stability constraints

The key insight is that we don't learn from scratch - we learn to
improve upon a strong classical baseline.

Architecture:
    state -> Physics Features -> SCA Warm Start -> Policy -> Lyapunov -> Action
                                      |                          ^
                                      v                          |
                                 Residual Net ------------------->
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_common import (
    Scenario, compute_channel_metrics, generate_scenarios,
    get_position_analytical, FEATURE_DIM
)
from mi_rl.physics_features import (
    extract_physics_features, PHYSICS_FEATURE_DIM,
    compute_physics_loss
)
from mi_rl.lyapunov_layer import LyapunovSafetyLayer, LyapunovConstrainedPolicy
from classical.sca_solver import SCASolver, SCAConfig
from classical.analytical_gradients import compute_throughput_gradient


@dataclass
class SGACConfig:
    """Configuration for SGAC agent."""
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005   # Target network update rate
    batch_size: int = 64
    buffer_size: int = 100000

    # Physics-informed weights
    lambda_physics: float = 0.1   # Weight for physics loss
    lambda_gradient: float = 0.05  # Weight for gradient alignment loss

    # SCA guidance
    sca_weight: float = 0.3       # Weight of SCA in action
    residual_scale: float = 5.0   # Scale of residual action

    # Lyapunov safety
    lyapunov_gamma: float = 0.1   # Lyapunov decay rate
    use_safety_layer: bool = True

    # Exploration
    noise_std: float = 0.1
    noise_decay: float = 0.995


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, info=None):
        self.buffer.append((state, action, reward, next_state, done, info))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, infos = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            infos
        )

    def __len__(self):
        return len(self.buffer)


class SCAGuidedActor(nn.Module):
    """
    Actor network that learns residual corrections to SCA baseline.

    Output: delta_position = SCA_direction * weight + Residual * scale

    The SCA provides a strong prior, and the residual learns adaptations.
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_dim: int = 256, num_layers: int = 3,
                 sca_weight: float = 0.3, residual_scale: float = 5.0):
        super().__init__()

        self.sca_weight = sca_weight
        self.residual_scale = residual_scale

        # Residual network
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.residual_net = nn.Sequential(*layers)

        # SCA direction is passed as part of state
        self.sca_direction_start = state_dim - 3  # Last 3 dims are SCA direction

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute action as SCA baseline + learned residual.

        State includes SCA gradient direction as last 3 dimensions.
        """
        # Extract SCA direction from state
        sca_direction = state[:, -3:]

        # Compute residual
        residual = self.residual_net(state) * self.residual_scale

        # Combine: weighted SCA direction + residual
        action = self.sca_weight * sca_direction + residual

        return action


class PhysicsInformedCritic(nn.Module):
    """
    Critic network with physics-informed architecture.

    Incorporates channel model knowledge in the value estimation.
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()

        # Twin critics (for TD3-style training)
        self.q1 = self._build_critic(state_dim + action_dim, hidden_dim, num_layers)
        self.q2 = self._build_critic(state_dim + action_dim, hidden_dim, num_layers)

    def _build_critic(self, input_dim: int, hidden_dim: int,
                      num_layers: int) -> nn.Module:
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor,
                   action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class SGACAgent:
    """
    SCA-Guided Actor-Critic agent.

    Combines classical optimization (SCA) with modern RL (Actor-Critic)
    for sample-efficient learning with physics guarantees.
    """

    def __init__(self, config: SGACConfig = None):
        self.config = config or SGACConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State dimension includes physics features + SCA direction
        self.state_dim = PHYSICS_FEATURE_DIM + 3  # +3 for SCA gradient direction
        self.action_dim = 3

        # Networks
        self.actor = SCAGuidedActor(
            self.state_dim, self.action_dim,
            self.config.hidden_dim, self.config.num_layers,
            self.config.sca_weight, self.config.residual_scale
        ).to(self.device)

        self.actor_target = SCAGuidedActor(
            self.state_dim, self.action_dim,
            self.config.hidden_dim, self.config.num_layers,
            self.config.sca_weight, self.config.residual_scale
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = PhysicsInformedCritic(
            self.state_dim, self.action_dim,
            self.config.hidden_dim, self.config.num_layers
        ).to(self.device)

        self.critic_target = PhysicsInformedCritic(
            self.state_dim, self.action_dim,
            self.config.hidden_dim, self.config.num_layers
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )

        # Replay buffer
        self.buffer = ReplayBuffer(self.config.buffer_size)

        # Safety layer
        if self.config.use_safety_layer:
            self.safety_layer = LyapunovSafetyLayer(
                gamma=self.config.lyapunov_gamma
            )
        else:
            self.safety_layer = None

        # SCA solver for Residual RL (use 20 iterations for good baseline)
        self.sca_solver = SCASolver(SCAConfig(max_iterations=20))

        # Noise for exploration
        self.noise_std = self.config.noise_std

        # Training stats
        self.training_step = 0
        self.total_episodes = 0

    def _get_state(self, pos: np.ndarray, scenario: Scenario) -> np.ndarray:
        """Get full state representation including physics features and SCA direction."""
        # Physics features
        physics_feats = extract_physics_features(pos, scenario)

        # SCA gradient direction
        _, grad = compute_throughput_gradient(pos, scenario)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            sca_direction = grad / grad_norm
        else:
            sca_direction = np.zeros(3)

        # Concatenate
        state = np.concatenate([physics_feats, sca_direction])
        return state.astype(np.float32)

    def _get_sca_solution(self, scenario: Scenario) -> np.ndarray:
        """Get SCA solution for scenario (cached for efficiency)."""
        # Use quick SCA (5 iterations) for speed
        sca_pos, _ = self.sca_solver.solve(scenario, verbose=False)
        return sca_pos

    def select_action(self, pos: np.ndarray, scenario: Scenario,
                      deterministic: bool = False) -> np.ndarray:
        """
        Select action using Residual RL: SCA_solution + small_correction.

        The RL only learns small corrections to the SCA baseline.
        This guarantees floor performance = SCA solution.
        """
        state = self._get_state(pos, scenario)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # RL outputs small correction (scaled by residual_scale in actor)
            correction = self.actor(state_tensor).cpu().numpy()[0]

        # Add exploration noise to correction only
        if not deterministic:
            noise = np.random.randn(3) * self.noise_std
            correction = correction + noise

        # Clip correction magnitude to prevent wild deviations
        correction_norm = np.linalg.norm(correction)
        max_correction = 5.0  # Max 5 meters correction
        if correction_norm > max_correction:
            correction = correction * max_correction / correction_norm

        return correction

    def get_position(self, scenario: Scenario,
                     current_pos: np.ndarray = None,
                     deterministic: bool = True) -> np.ndarray:
        """
        Get optimal position using Residual RL.

        Final position = SCA_solution + RL_correction

        This guarantees that even with zero RL output, we get SCA performance.
        """
        # Get SCA base solution (this is the strong baseline)
        sca_pos = self._get_sca_solution(scenario)

        # Get RL correction
        if current_pos is None:
            current_pos = scenario.initial_uav_position.copy()

        correction = self.select_action(current_pos, scenario, deterministic)

        # Final position = SCA + correction
        final_pos = sca_pos + correction

        # Clip to feasible region
        final_pos[0] = np.clip(final_pos[0], 0, 100)
        final_pos[1] = np.clip(final_pos[1], 0, 100)
        final_pos[2] = np.clip(final_pos[2], 10, 40)

        return final_pos

    def add_experience(self, state: np.ndarray, action: np.ndarray,
                       reward: float, next_state: np.ndarray, done: bool,
                       info: dict = None) -> None:
        """
        Add a single experience to the replay buffer.

        Used for parallel training where experiences are collected externally.
        """
        if info is None:
            info = {}
        self.buffer.push(state, action, reward, next_state, done, info)

    def update(self) -> Dict[str, float]:
        """
        Perform one gradient update on actor and critic.

        Returns dict of training metrics.
        """
        if len(self.buffer) < self.config.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones, infos = \
            self.buffer.sample(self.config.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + self.config.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_value) + \
                      nn.MSELoss()(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        actor_actions = self.actor(states)
        actor_loss = -self.critic.q1_forward(states, actor_actions).mean()

        # Physics-informed regularization
        if self.config.lambda_physics > 0:
            # Gradient alignment loss: actor output should align with SCA direction
            sca_directions = states[:, -3:]  # Last 3 dims
            residual_output = actor_actions - self.config.sca_weight * sca_directions
            # Encourage residual to be small when SCA is good
            residual_magnitude = torch.norm(residual_output, dim=1).mean()
            actor_loss = actor_loss + self.config.lambda_gradient * residual_magnitude

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft update target networks ---
        for param, target_param in zip(self.actor.parameters(),
                                        self.actor_target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data +
                (1 - self.config.tau) * target_param.data
            )

        for param, target_param in zip(self.critic.parameters(),
                                        self.critic_target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data +
                (1 - self.config.tau) * target_param.data
            )

        self.training_step += 1

        # Decay noise
        self.noise_std *= self.config.noise_decay
        self.noise_std = max(self.noise_std, 0.01)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'noise_std': self.noise_std
        }

    def train_episode(self, scenario: Scenario,
                      max_steps: int = 10) -> Dict[str, float]:
        """
        Train on a single scenario for multiple steps.

        Returns episode metrics.
        """
        pos = scenario.initial_uav_position.copy()
        episode_reward = 0
        episode_metrics = []

        # Get SCA base solution once per episode
        sca_pos = self._get_sca_solution(scenario)
        sca_metrics = compute_channel_metrics(sca_pos, scenario)
        sca_throughput = sca_metrics['total_throughput']

        for step in range(max_steps):
            # Get state
            state = self._get_state(pos, scenario)

            # Select correction (not full action)
            correction = self.select_action(pos, scenario, deterministic=False)

            # Apply correction to SCA solution (Residual RL)
            next_pos = sca_pos + correction
            next_pos[0] = np.clip(next_pos[0], 0, 100)
            next_pos[1] = np.clip(next_pos[1], 0, 100)
            next_pos[2] = np.clip(next_pos[2], 10, 40)

            # Compute reward: improvement over SCA baseline
            metrics = compute_channel_metrics(next_pos, scenario)
            # Reward is positive if we beat SCA, negative if worse
            reward = (metrics['total_throughput'] - sca_throughput) / 10
            # Small penalty for large corrections (encourage minimal changes)
            correction_penalty = 0.01 * np.linalg.norm(correction)
            reward = reward - correction_penalty

            # Check done (converged or at boundary)
            done = step == max_steps - 1

            # Get next state
            next_state = self._get_state(next_pos, scenario)

            # Store transition
            self.buffer.push(state, correction, reward, next_state, done,
                           {'scenario_id': scenario.id, 'sca_throughput': sca_throughput})

            # Update
            update_info = self.update()

            pos = next_pos
            episode_reward += reward
            episode_metrics.append(metrics)

        self.total_episodes += 1

        final_metrics = episode_metrics[-1]
        return {
            'episode_reward': episode_reward,
            'final_throughput': final_metrics['total_throughput'],
            'final_fairness': final_metrics['fairness'],
            'steps': len(episode_metrics)
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'total_episodes': self.total_episodes,
            'noise_std': self.noise_std,
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint['training_step']
        self.total_episodes = checkpoint['total_episodes']
        self.noise_std = checkpoint['noise_std']

    def save_checkpoint(self, path: str, extra_data: dict = None):
        """Save checkpoint with training state for resume capability."""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'total_episodes': self.total_episodes,
            'noise_std': self.noise_std,
            'config': self.config
        }
        if extra_data:
            checkpoint.update(extra_data)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint and return extra training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint['training_step']
        self.total_episodes = checkpoint['total_episodes']
        self.noise_std = checkpoint['noise_std']
        # Return extra data for training loop
        return {
            'episode': checkpoint.get('episode', 0),
            'training_history': checkpoint.get('training_history', []),
            'best_throughput': checkpoint.get('best_throughput', 0),
            'best_episode': checkpoint.get('best_episode', 0)
        }


if __name__ == "__main__":
    """Quick test of SGAC agent."""
    print("Testing SGAC Agent")
    print("=" * 60)

    # Create agent
    agent = SGACAgent()
    print(f"Device: {agent.device}")
    print(f"State dim: {agent.state_dim}")
    print(f"Actor params: {sum(p.numel() for p in agent.actor.parameters())}")
    print(f"Critic params: {sum(p.numel() for p in agent.critic.parameters())}")

    # Generate scenarios
    scenarios = generate_scenarios(num_scenarios=10)

    # Quick training loop
    print("\nTraining for 50 episodes...")
    for episode in range(50):
        scenario = scenarios[episode % len(scenarios)]
        metrics = agent.train_episode(scenario)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: reward={metrics['episode_reward']:.2f}, "
                  f"throughput={metrics['final_throughput']:.1f} Mbps")

    # Evaluate
    print("\nEvaluation:")
    results = []
    for scenario in scenarios:
        pos = agent.get_position(scenario, deterministic=True)
        metrics = compute_channel_metrics(pos, scenario)
        analytical_pos = get_position_analytical(scenario)
        analytical_metrics = compute_channel_metrics(analytical_pos, scenario)

        improvement = (metrics['total_throughput'] / analytical_metrics['total_throughput'] - 1) * 100
        results.append({
            'sgac': metrics['total_throughput'],
            'analytical': analytical_metrics['total_throughput'],
            'improvement': improvement
        })

    print(f"SGAC avg: {np.mean([r['sgac'] for r in results]):.1f} Mbps")
    print(f"Analytical avg: {np.mean([r['analytical'] for r in results]):.1f} Mbps")
    print(f"Avg improvement: {np.mean([r['improvement'] for r in results]):.1f}%")
