#!/usr/bin/env python3
"""
Train RL Baselines (DDPG, TD3, SAC, PPO) for comparison with SGAC.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import time

from environment import UAVRelayEnv

from ddpg_agent import DDPGAgent
from td3_agent import TD3Agent
from sac_agent import SACAgent
from ppo_agent import PPOAgent


def train_offpolicy(agent, env, config, agent_name):
    """Train off-policy agents (DDPG, TD3, SAC)."""
    history = {
        'episode': [],
        'episode_reward': [],
        'episode_throughput': [],
        'critic_loss': [],
        'actor_loss': [],
    }

    best_throughput = -float('inf')
    exploration_noise = config['exploration_start']
    noise_decay = (config['exploration_start'] - config['exploration_end']) / config['episodes']

    print(f"\nTraining {agent_name} for {config['episodes']} episodes...")
    start_time = time.time()

    for episode in range(1, config['episodes'] + 1):
        state = env.reset()
        episode_reward = 0
        episode_throughputs = []

        for step in range(config['max_steps']):
            # Select action
            if agent_name == 'SAC':
                action = agent.select_action(state, deterministic=False)
            else:
                action = agent.select_action(state, noise=exploration_noise)

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            if episode > config['warmup_episodes']:
                critic_loss, actor_loss = agent.train_step()

            episode_reward += reward
            episode_throughputs.append(info['throughput'])
            state = next_state

            if done:
                break

        # Decay exploration
        exploration_noise = max(config['exploration_end'], exploration_noise - noise_decay)

        # Record history
        avg_throughput = np.mean(episode_throughputs)
        history['episode'].append(episode)
        history['episode_reward'].append(episode_reward)
        history['episode_throughput'].append(avg_throughput)
        history['critic_loss'].append(agent.critic_loss)
        history['actor_loss'].append(agent.actor_loss)

        if avg_throughput > best_throughput:
            best_throughput = avg_throughput
            agent.save(config['output_dir'] / f'{agent_name.lower()}_best.pt')

        if episode % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Episode {episode}/{config['episodes']} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Throughput: {avg_throughput:.2f} Mbps | "
                  f"Best: {best_throughput:.2f} Mbps | "
                  f"Time: {elapsed:.1f}s")

    return history


def train_ppo(agent, env, config):
    """Train PPO agent."""
    history = {
        'episode': [],
        'episode_reward': [],
        'episode_throughput': [],
        'critic_loss': [],
        'actor_loss': [],
    }

    best_throughput = -float('inf')
    rollout_steps = config.get('rollout_steps', 2048)

    print(f"\nTraining PPO for {config['episodes']} episodes...")
    start_time = time.time()

    total_steps = 0
    episode = 0

    while episode < config['episodes']:
        # Collect rollout
        state = env.reset()
        episode_reward = 0
        episode_throughputs = []
        steps_in_episode = 0

        for _ in range(rollout_steps):
            action, log_prob, value = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_throughputs.append(info['throughput'])
            state = next_state
            steps_in_episode += 1
            total_steps += 1

            if done:
                # Record episode
                episode += 1
                avg_throughput = np.mean(episode_throughputs)

                history['episode'].append(episode)
                history['episode_reward'].append(episode_reward)
                history['episode_throughput'].append(avg_throughput)
                history['critic_loss'].append(agent.critic_loss)
                history['actor_loss'].append(agent.actor_loss)

                if avg_throughput > best_throughput:
                    best_throughput = avg_throughput
                    agent.save(config['output_dir'] / 'ppo_best.pt')

                if episode % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Episode {episode}/{config['episodes']} | "
                          f"Reward: {episode_reward:.2f} | "
                          f"Throughput: {avg_throughput:.2f} Mbps | "
                          f"Best: {best_throughput:.2f} Mbps | "
                          f"Time: {elapsed:.1f}s")

                if episode >= config['episodes']:
                    break

                # Reset for next episode
                state = env.reset()
                episode_reward = 0
                episode_throughputs = []

        # PPO update
        critic_loss, actor_loss = agent.train_step()

    return history


def evaluate_agent(agent, env, num_episodes=50, agent_name='Agent'):
    """Evaluate trained agent."""
    throughputs = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_throughputs = []

        for _ in range(50):
            if agent_name == 'PPO':
                action, _, _ = agent.select_action(state, deterministic=True)
            elif agent_name == 'SAC':
                action = agent.select_action(state, deterministic=True)
            else:
                action = agent.select_action(state, noise=0)

            next_state, reward, done, info = env.step(action)
            episode_throughputs.append(info['throughput'])
            state = next_state

            if done:
                break

        throughputs.append(np.mean(episode_throughputs))

    return {
        'mean': np.mean(throughputs),
        'std': np.std(throughputs),
        'min': np.min(throughputs),
        'max': np.max(throughputs)
    }


def main():
    parser = argparse.ArgumentParser(description='Train RL Baselines')
    parser.add_argument('--agent', type=str, default='all',
                        choices=['ddpg', 'td3', 'sac', 'ppo', 'all'],
                        help='Agent to train')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--num_users', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./results_baselines')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'episodes': args.episodes,
        'max_steps': args.max_steps,
        'warmup_episodes': 50,
        'exploration_start': 0.3,
        'exploration_end': 0.05,
        'output_dir': output_dir,
    }

    # Environment
    env = UAVRelayEnv(num_users=args.num_users, max_steps=args.max_steps, seed=args.seed)

    # Results storage
    all_results = {}

    agents_to_train = ['ddpg', 'td3', 'sac', 'ppo'] if args.agent == 'all' else [args.agent]

    for agent_name in agents_to_train:
        print(f"\n{'='*60}")
        print(f"Training {agent_name.upper()}")
        print('='*60)

        # Create agent
        if agent_name == 'ddpg':
            agent = DDPGAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=256,
                lr_actor=1e-4,
                lr_critic=1e-3,
            )
            history = train_offpolicy(agent, env, config, 'DDPG')

        elif agent_name == 'td3':
            agent = TD3Agent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=256,
            )
            history = train_offpolicy(agent, env, config, 'TD3')

        elif agent_name == 'sac':
            agent = SACAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=256,
            )
            history = train_offpolicy(agent, env, config, 'SAC')

        elif agent_name == 'ppo':
            agent = PPOAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=256,
            )
            history = train_ppo(agent, env, config)

        # Load best model for evaluation
        agent.load(output_dir / f'{agent_name}_best.pt')

        # Evaluate
        print(f"\nEvaluating {agent_name.upper()}...")
        eval_results = evaluate_agent(agent, env, num_episodes=50, agent_name=agent_name.upper())

        print(f"  Mean Throughput: {eval_results['mean']:.2f} +/- {eval_results['std']:.2f} Mbps")
        print(f"  Min: {eval_results['min']:.2f} | Max: {eval_results['max']:.2f} Mbps")

        # Save results
        all_results[agent_name] = {
            'history': history,
            'evaluation': eval_results
        }

        # Save history
        with open(output_dir / f'{agent_name}_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # Save combined results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'results': {name: data['evaluation'] for name, data in all_results.items()}
    }

    with open(output_dir / 'baseline_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("BASELINE COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Method':<10} {'Mean (Mbps)':<15} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*60)
    for name, data in all_results.items():
        r = data['evaluation']
        print(f"{name.upper():<10} {r['mean']:<15.2f} {r['std']:<10.2f} {r['min']:<10.2f} {r['max']:<10.2f}")
    print("="*60)


if __name__ == '__main__':
    main()
