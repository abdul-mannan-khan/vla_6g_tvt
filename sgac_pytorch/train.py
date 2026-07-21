#!/usr/bin/env python3
"""
SGAC Training Script for UAV Relay Positioning

This is the main training script that:
1. Creates environments and scenarios
2. Trains the SGAC agent
3. Evaluates and logs performance
4. Saves checkpoints and results
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import UAVRelayEnv, ScenarioGenerator, ChannelParams
from sgac_agent import SGACAgent, BaselineAgents


def parse_args():
    parser = argparse.ArgumentParser(description='Train SGAC for UAV Relay Positioning')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Max steps per episode')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Evaluate every N episodes')
    parser.add_argument('--save-interval', type=int, default=500,
                        help='Save checkpoint every N episodes')

    # Environment parameters
    parser.add_argument('--num-users', type=int, default=5,
                        help='Number of IoT users')
    parser.add_argument('--num-scenarios', type=int, default=100,
                        help='Number of training scenarios')
    parser.add_argument('--num-eval-scenarios', type=int, default=50,
                        help='Number of evaluation scenarios')

    # SGAC parameters
    parser.add_argument('--alpha-sca', type=float, default=0.7,
                        help='SCA guidance weight')
    parser.add_argument('--beta-nn', type=float, default=0.3,
                        help='Neural network weight')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer dimension')

    # Learning parameters
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                        help='Actor learning rate')
    parser.add_argument('--lr-critic', type=float, default=3e-4,
                        help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='Replay buffer size')

    # Exploration
    parser.add_argument('--exploration-start', type=float, default=0.3,
                        help='Initial exploration noise')
    parser.add_argument('--exploration-end', type=float, default=0.05,
                        help='Final exploration noise')
    parser.add_argument('--warmup-episodes', type=int, default=50,
                        help='Episodes before training starts')

    # Paths
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_agent(
    agent: SGACAgent,
    env: UAVRelayEnv,
    scenarios: List[np.ndarray],
    num_steps: int = 50
) -> Dict[str, float]:
    """
    Evaluate agent on a set of scenarios.

    Returns:
        Dict with evaluation metrics
    """
    results = {
        'throughputs': [],
        'sca_throughputs': [],
        'improvements': [],
        'improvement_pcts': [],
        'floor_activations': [],
        'final_deviations': []
    }

    for user_positions in scenarios:
        state = env.reset(user_positions)
        sca_throughput = env.sca_throughput
        sca_direction = env.get_sca_action()

        floor_count = 0
        for _ in range(num_steps):
            action = agent.select_action(state, sca_direction, evaluate=True)
            next_state, reward, done, info = env.step(action)

            if info['floor_activated']:
                floor_count += 1

            state = next_state
            if done:
                break

        final_throughput = info['throughput']
        improvement = final_throughput - sca_throughput

        results['throughputs'].append(final_throughput / 1e6)  # Mbps
        results['sca_throughputs'].append(sca_throughput / 1e6)
        results['improvements'].append(improvement / 1e6)
        results['improvement_pcts'].append(100 * improvement / sca_throughput if sca_throughput > 0 else 0)
        results['floor_activations'].append(floor_count)
        results['final_deviations'].append(info['deviation'])

    # Aggregate statistics
    metrics = {
        'mean_throughput': np.mean(results['throughputs']),
        'std_throughput': np.std(results['throughputs']),
        'mean_sca_throughput': np.mean(results['sca_throughputs']),
        'mean_improvement': np.mean(results['improvements']),
        'mean_improvement_pct': np.mean(results['improvement_pcts']),
        'mean_floor_activations': np.mean(results['floor_activations']),
        'floor_rate': np.mean([f > 0 for f in results['floor_activations']]),
        'mean_deviation': np.mean(results['final_deviations']),
        'min_throughput': np.min(results['throughputs']),
        'max_throughput': np.max(results['throughputs']),
    }

    # Check floor guarantee (SGAC should never be worse than SCA)
    violations = sum(1 for t, s in zip(results['throughputs'], results['sca_throughputs'])
                     if t < s - 0.01)  # Small tolerance
    metrics['floor_violations'] = violations
    metrics['floor_guarantee_rate'] = 1.0 - violations / len(scenarios)

    return metrics


def evaluate_baselines(
    env: UAVRelayEnv,
    scenarios: List[np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Evaluate baseline methods."""
    results = {
        'random': {'throughputs': []},
        'analytical': {'throughputs': []},
        'sca': {'throughputs': []}
    }

    for user_positions in scenarios:
        env.reset(user_positions)

        # Random baseline
        random_pos = np.array([
            np.random.uniform(0, env.params.area_size),
            np.random.uniform(0, env.params.area_size),
            np.random.uniform(env.params.h_min, env.params.h_max)
        ])
        random_throughput = env._compute_throughput(random_pos)
        results['random']['throughputs'].append(random_throughput / 1e6)

        # Analytical baseline (centroid)
        centroid = np.mean(user_positions, axis=0)
        centroid[2] = (env.params.h_min + env.params.h_max) / 2
        analytical_throughput = env._compute_throughput(centroid)
        results['analytical']['throughputs'].append(analytical_throughput / 1e6)

        # SCA baseline
        results['sca']['throughputs'].append(env.sca_throughput / 1e6)

    # Compute statistics
    baselines = {}
    for name, data in results.items():
        baselines[name] = {
            'mean': np.mean(data['throughputs']),
            'std': np.std(data['throughputs']),
            'min': np.min(data['throughputs']),
            'max': np.max(data['throughputs'])
        }

    return baselines


def train(args):
    """Main training function."""
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'sgac_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("SGAC Training - UAV Relay Positioning")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Episodes: {args.episodes}")
    print(f"Scenarios: {args.num_scenarios} training, {args.num_eval_scenarios} eval")
    print()

    # Set seed
    set_seed(args.seed)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Generate scenarios
    print("Generating scenarios...")
    scenario_gen = ScenarioGenerator(
        num_users=args.num_users,
        seed=args.seed
    )
    train_scenarios = scenario_gen.generate(args.num_scenarios)
    eval_scenarios = scenario_gen.generate(args.num_eval_scenarios)

    # Create environment
    env = UAVRelayEnv(
        num_users=args.num_users,
        max_steps=args.max_steps,
        seed=args.seed
    )

    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")

    # Create agent
    agent = SGACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
        alpha_sca=args.alpha_sca,
        beta_nn=args.beta_nn,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device=device
    )

    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        agent.load(args.resume)
        # Try to extract episode number from filename
        try:
            start_episode = int(Path(args.resume).stem.split('_')[-1])
        except:
            pass

    # Evaluate baselines once
    print("\nEvaluating baselines...")
    baselines = evaluate_baselines(env, eval_scenarios)
    print(f"Random:     {baselines['random']['mean']:.2f} +/- {baselines['random']['std']:.2f} Mbps")
    print(f"Analytical: {baselines['analytical']['mean']:.2f} +/- {baselines['analytical']['std']:.2f} Mbps")
    print(f"SCA:        {baselines['sca']['mean']:.2f} +/- {baselines['sca']['std']:.2f} Mbps")

    # Training history
    history = {
        'episode': [],
        'episode_reward': [],
        'episode_throughput': [],
        'sca_throughput': [],
        'floor_activations': [],
        'critic_loss': [],
        'actor_loss': [],
        'eval_throughput': [],
        'eval_improvement_pct': [],
        'exploration_noise': []
    }

    # Training loop
    print("\nStarting training...")
    best_eval_throughput = 0
    training_start_time = time.time()

    pbar = tqdm(range(start_episode, args.episodes), desc="Training")

    for episode in pbar:
        # Anneal exploration noise
        progress = min(1.0, episode / (args.episodes * 0.8))
        exploration_noise = args.exploration_start + progress * (args.exploration_end - args.exploration_start)
        agent.set_exploration_noise(exploration_noise)

        # Sample scenario
        scenario_idx = np.random.randint(len(train_scenarios))
        user_positions = train_scenarios[scenario_idx]

        # Reset environment
        state = env.reset(user_positions)
        sca_direction = env.get_sca_action()

        episode_reward = 0
        episode_floor_count = 0
        critic_losses = []
        actor_losses = []

        # Episode loop
        for step in range(args.max_steps):
            # Select action
            action = agent.select_action(state, sca_direction, evaluate=False)

            # Environment step
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(
                state, action, reward, next_state, done, sca_direction
            )

            episode_reward += reward
            if info['floor_activated']:
                episode_floor_count += 1

            # Training step (after warmup)
            if episode >= args.warmup_episodes:
                train_info = agent.train_step()
                if train_info:
                    critic_losses.append(train_info.get('critic_loss', 0))
                    actor_losses.append(train_info.get('actor_loss', 0))

            state = next_state
            if done:
                break

        # Record episode stats
        history['episode'].append(episode)
        history['episode_reward'].append(episode_reward)
        history['episode_throughput'].append(info['throughput_mbps'])
        history['sca_throughput'].append(info['sca_throughput_mbps'])
        history['floor_activations'].append(episode_floor_count)
        history['critic_loss'].append(np.mean(critic_losses) if critic_losses else 0)
        history['actor_loss'].append(np.mean(actor_losses) if actor_losses else 0)
        history['exploration_noise'].append(exploration_noise)

        # Update progress bar
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'throughput': f'{info["throughput_mbps"]:.1f}',
            'floor': episode_floor_count,
            'noise': f'{exploration_noise:.3f}'
        })

        # Evaluation
        if (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_agent(agent, env, eval_scenarios, args.max_steps)

            history['eval_throughput'].append(eval_metrics['mean_throughput'])
            history['eval_improvement_pct'].append(eval_metrics['mean_improvement_pct'])

            print(f"\n[Episode {episode + 1}] Evaluation:")
            print(f"  Throughput: {eval_metrics['mean_throughput']:.2f} +/- {eval_metrics['std_throughput']:.2f} Mbps")
            print(f"  vs SCA: {eval_metrics['mean_improvement_pct']:+.2f}%")
            print(f"  Floor violations: {eval_metrics['floor_violations']}/{args.num_eval_scenarios}")
            print(f"  Floor activation rate: {eval_metrics['mean_floor_activations']:.1f}/{args.max_steps}")

            # Save best model
            if eval_metrics['mean_throughput'] > best_eval_throughput:
                best_eval_throughput = eval_metrics['mean_throughput']
                agent.save(output_dir / 'best_model.pt')
                print(f"  New best model saved! ({best_eval_throughput:.2f} Mbps)")

        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            agent.save(output_dir / f'checkpoint_{episode + 1}.pt')

            # Save history
            with open(output_dir / 'history.json', 'w') as f:
                json.dump(history, f)

    # Training complete
    training_time = time.time() - training_start_time
    print(f"\nTraining completed in {training_time / 60:.1f} minutes")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    final_metrics = evaluate_agent(agent, env, eval_scenarios, args.max_steps)

    print(f"\nSGAC Performance:")
    print(f"  Mean Throughput: {final_metrics['mean_throughput']:.2f} +/- {final_metrics['std_throughput']:.2f} Mbps")
    print(f"  Min/Max: {final_metrics['min_throughput']:.2f} / {final_metrics['max_throughput']:.2f} Mbps")
    print(f"  Improvement vs SCA: {final_metrics['mean_improvement_pct']:+.2f}%")
    print(f"  Floor Guarantee: {final_metrics['floor_guarantee_rate'] * 100:.1f}%")

    # Comparison with baselines
    print("\nComparison with Baselines:")
    for name, stats in baselines.items():
        improvement = (final_metrics['mean_throughput'] - stats['mean']) / stats['mean'] * 100
        print(f"  vs {name.capitalize()}: {improvement:+.1f}%")

    # Save final results
    final_results = {
        'sgac': final_metrics,
        'baselines': baselines,
        'training_time_minutes': training_time / 60,
        'total_episodes': args.episodes,
        'best_throughput': best_eval_throughput
    }

    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # Save final model
    agent.save(output_dir / 'final_model.pt')

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    print(f"\nResults saved to: {output_dir}")

    return final_metrics, baselines


if __name__ == '__main__':
    args = parse_args()
    train(args)
