#!/usr/bin/env python3
"""
SGAC Training Script - FIXED VERSION

Key fixes:
1. Uses fixed environment with proper reward signal
2. Curriculum learning: gradually shift from SCA to learned
3. Proper evaluation with floor guarantee
4. Logging to verify actual learning
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment_fixed import UAVRelayEnvFixed, ScenarioGenerator
from sgac_agent_fixed import SGACAgentFixed


def parse_args():
    parser = argparse.ArgumentParser(description='Train FIXED SGAC')

    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)

    parser.add_argument('--num-users', type=int, default=5)
    parser.add_argument('--num-scenarios', type=int, default=100)
    parser.add_argument('--num-eval-scenarios', type=int, default=50)

    # FIXED: Better default weights
    parser.add_argument('--alpha-sca', type=float, default=0.5)
    parser.add_argument('--beta-nn', type=float, default=0.5)
    parser.add_argument('--hidden-dim', type=int, default=256)

    # Curriculum learning
    parser.add_argument('--use-curriculum', action='store_true', default=True)
    parser.add_argument('--curriculum-start', type=float, default=0.7,
                        help='Initial SCA weight')
    parser.add_argument('--curriculum-end', type=float, default=0.3,
                        help='Final SCA weight')

    # Start perturbation
    parser.add_argument('--start-perturbation', type=float, default=10.0,
                        help='How far from SCA to start (meters)')

    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=1000000)

    parser.add_argument('--exploration-start', type=float, default=0.3)
    parser.add_argument('--exploration-end', type=float, default=0.05)
    parser.add_argument('--warmup-episodes', type=int, default=50)

    parser.add_argument('--output-dir', type=str, default='./results_fixed')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(agent, env, scenarios, num_steps=50) -> Dict:
    """Evaluate with floor guarantee enabled."""
    env.set_training_mode(False)  # Enable floor for safety

    results = {
        'throughputs': [],
        'sca_throughputs': [],
        'improvements': [],
        'floor_activations': [],
    }

    for user_positions in scenarios:
        state = env.reset(user_positions)
        sca_direction = env.get_sca_action()

        floor_count = 0
        for _ in range(num_steps):
            action = agent.select_action(state, sca_direction, evaluate=True)
            next_state, _, done, info = env.step(action)

            if info['floor_activated']:
                floor_count += 1

            state = next_state
            if done:
                break

        results['throughputs'].append(info['throughput'] / 1e6)
        results['sca_throughputs'].append(info['sca_throughput'] / 1e6)
        results['improvements'].append((info['throughput'] - info['sca_throughput']) / 1e6)
        results['floor_activations'].append(floor_count)

    env.set_training_mode(True)  # Back to training mode

    metrics = {
        'mean_throughput': np.mean(results['throughputs']),
        'std_throughput': np.std(results['throughputs']),
        'mean_sca_throughput': np.mean(results['sca_throughputs']),
        'mean_improvement': np.mean(results['improvements']),
        'mean_improvement_pct': np.mean(results['improvements']) / np.mean(results['sca_throughputs']) * 100,
        'mean_floor_activations': np.mean(results['floor_activations']),
        'floor_rate': np.mean([f > 0 for f in results['floor_activations']]),
        'min_throughput': np.min(results['throughputs']),
        'max_throughput': np.max(results['throughputs']),
    }

    # Floor guarantee check
    violations = sum(1 for t, s in zip(results['throughputs'], results['sca_throughputs'])
                     if t < s - 0.01)
    metrics['floor_violations'] = violations
    metrics['floor_guarantee_rate'] = 1.0 - violations / len(scenarios)

    return metrics


def train(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'sgac_fixed_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("FIXED SGAC Training - UAV Relay Positioning")
    print("=" * 60)
    print(f"Key fixes applied:")
    print(f"  - Training mode: No floor, negative rewards allowed")
    print(f"  - Start perturbation: {args.start_perturbation}m from SCA")
    print(f"  - Curriculum learning: alpha {args.curriculum_start} -> {args.curriculum_end}")
    print(f"Output: {output_dir}")
    print()

    set_seed(args.seed)

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if args.device == 'auto' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"Device: {device}")

    # Generate scenarios
    scenario_gen = ScenarioGenerator(num_users=args.num_users, seed=args.seed)
    train_scenarios = scenario_gen.generate(args.num_scenarios)
    eval_scenarios = scenario_gen.generate(args.num_eval_scenarios)

    # Create FIXED environment
    env = UAVRelayEnvFixed(
        num_users=args.num_users,
        max_steps=args.max_steps,
        seed=args.seed,
        training_mode=True,
        start_perturbation=args.start_perturbation
    )

    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")

    # Create FIXED agent
    agent = SGACAgentFixed(
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

    # Training history with VERIFICATION metrics
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
        'exploration_noise': [],
        'alpha_sca': [],  # Track curriculum
        'beta_nn': [],
        'negative_reward_count': [],  # VERIFICATION: should be > 0
    }

    print("\nStarting training...")
    best_eval_throughput = 0
    start_time = time.time()

    pbar = tqdm(range(args.episodes), desc="Training")

    for episode in pbar:
        # Curriculum learning: shift from SCA to learned
        if args.use_curriculum:
            progress = min(1.0, episode / (args.episodes * 0.8))
            alpha = args.curriculum_start - progress * (args.curriculum_start - args.curriculum_end)
            beta = 1.0 - alpha
            agent.policy.set_mixing_weights(alpha, beta)
        else:
            alpha, beta = args.alpha_sca, args.beta_nn

        # Exploration annealing
        progress = min(1.0, episode / (args.episodes * 0.8))
        exploration = args.exploration_start + progress * (args.exploration_end - args.exploration_start)
        agent.set_exploration_noise(exploration)

        # Sample scenario
        scenario_idx = np.random.randint(len(train_scenarios))
        state = env.reset(train_scenarios[scenario_idx])
        sca_direction = env.get_sca_action()

        episode_reward = 0
        episode_floor_count = 0
        negative_rewards = 0
        critic_losses = []
        actor_losses = []

        for step in range(args.max_steps):
            action = agent.select_action(state, sca_direction, evaluate=False)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done, sca_direction)

            episode_reward += reward
            if reward < 0:
                negative_rewards += 1
            if info.get('floor_activated', False):
                episode_floor_count += 1

            if episode >= args.warmup_episodes:
                train_info = agent.train_step()
                if train_info:
                    critic_losses.append(train_info.get('critic_loss', 0))
                    actor_losses.append(train_info.get('actor_loss', 0))

            state = next_state
            if done:
                break

        # Record history
        history['episode'].append(episode)
        history['episode_reward'].append(episode_reward)
        history['episode_throughput'].append(info['throughput_mbps'])
        history['sca_throughput'].append(info['sca_throughput_mbps'])
        history['floor_activations'].append(episode_floor_count)
        history['critic_loss'].append(np.mean(critic_losses) if critic_losses else 0)
        history['actor_loss'].append(np.mean(actor_losses) if actor_losses else 0)
        history['exploration_noise'].append(exploration)
        history['alpha_sca'].append(alpha)
        history['beta_nn'].append(beta)
        history['negative_reward_count'].append(negative_rewards)

        pbar.set_postfix({
            'R': f'{episode_reward:.1f}',
            'T': f'{info["throughput_mbps"]:.0f}',
            'neg': negative_rewards,
            'α': f'{alpha:.2f}'
        })

        # Evaluation
        if (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_agent(agent, env, eval_scenarios, args.max_steps)

            history['eval_throughput'].append(eval_metrics['mean_throughput'])
            history['eval_improvement_pct'].append(eval_metrics['mean_improvement_pct'])

            print(f"\n[Ep {episode + 1}] Eval: {eval_metrics['mean_throughput']:.2f} Mbps "
                  f"(+{eval_metrics['mean_improvement_pct']:.2f}% vs SCA), "
                  f"Floor viol: {eval_metrics['floor_violations']}")

            # VERIFICATION: Check learning progress
            if episode >= 500:
                early_rewards = np.mean(history['episode_reward'][:100])
                recent_rewards = np.mean(history['episode_reward'][-100:])
                improvement = (recent_rewards - early_rewards) / abs(early_rewards) * 100
                print(f"      Learning check: Early R={early_rewards:.1f} -> Recent R={recent_rewards:.1f} "
                      f"({improvement:+.1f}%)")

            if eval_metrics['mean_throughput'] > best_eval_throughput:
                best_eval_throughput = eval_metrics['mean_throughput']
                agent.save(output_dir / 'best_model.pt')

        if (episode + 1) % args.save_interval == 0:
            agent.save(output_dir / f'checkpoint_{episode + 1}.pt')
            with open(output_dir / 'history.json', 'w') as f:
                json.dump(history, f)

    # Final evaluation
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.1f} minutes")

    print("\n" + "=" * 60)
    print("VERIFICATION: Did the agent actually learn?")
    print("=" * 60)

    early_rewards = np.mean(history['episode_reward'][:200])
    late_rewards = np.mean(history['episode_reward'][-200:])
    reward_improvement = (late_rewards - early_rewards) / abs(early_rewards) * 100

    early_neg = np.mean(history['negative_reward_count'][:200])
    late_neg = np.mean(history['negative_reward_count'][-200:])

    print(f"Reward: {early_rewards:.1f} -> {late_rewards:.1f} ({reward_improvement:+.1f}%)")
    print(f"Negative rewards per ep: {early_neg:.1f} -> {late_neg:.1f}")

    if reward_improvement > 10:
        print("✓ LEARNING DETECTED: Significant reward improvement")
    else:
        print("✗ WARNING: Minimal learning detected")

    # Final eval
    final_metrics = evaluate_agent(agent, env, eval_scenarios, args.max_steps)

    print(f"\nFinal Performance:")
    print(f"  Throughput: {final_metrics['mean_throughput']:.2f} ± {final_metrics['std_throughput']:.2f} Mbps")
    print(f"  vs SCA: {final_metrics['mean_improvement_pct']:+.2f}%")
    print(f"  Floor guarantee: {final_metrics['floor_guarantee_rate'] * 100:.1f}%")

    # Save results
    final_results = {
        'sgac': final_metrics,
        'training_time_minutes': training_time / 60,
        'total_episodes': args.episodes,
        'best_throughput': best_eval_throughput,
        'learning_verification': {
            'early_reward': early_rewards,
            'late_reward': late_rewards,
            'improvement_pct': reward_improvement,
            'learning_detected': reward_improvement > 10
        }
    }

    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    agent.save(output_dir / 'final_model.pt')

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    print(f"\nResults saved to: {output_dir}")

    return final_metrics


if __name__ == '__main__':
    args = parse_args()
    train(args)
