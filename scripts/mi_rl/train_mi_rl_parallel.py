#!/usr/bin/env python3
"""
Parallelized Training Script for Math-Informed RL (MI-RL).

Uses multiprocessing to parallelize episode collection across multiple CPUs,
while keeping neural network updates on the main thread.

This achieves better CPU utilization since SCA solving is CPU-bound.

Usage:
    python train_mi_rl_parallel.py --episodes 50000 --workers 8
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_common import (
    generate_scenarios, compute_channel_metrics,
    get_position_analytical, get_position_random, get_position_static
)
from mi_rl.sgac_agent import SGACAgent, SGACConfig
from mi_rl.physics_features import extract_physics_features
from classical.sca_solver import SCASolver, SCAConfig
from classical.analytical_gradients import compute_throughput_gradient


def collect_experience_worker(args):
    """
    Worker function to collect experience from a single episode.

    Runs in a separate process to parallelize SCA solving.

    Args:
        args: Tuple of (scenario_dict, sca_iterations, residual_scale)

    Returns:
        List of experience tuples for replay buffer
    """
    scenario_dict, sca_iterations, residual_scale = args

    # Reconstruct scenario from dict (can't pickle Scenario directly)
    from eval_common import Scenario
    scenario = Scenario(
        id=scenario_dict['id'],
        bs_position=np.array(scenario_dict['bs_position']),
        user_positions=[np.array(p) for p in scenario_dict['user_positions']],
        user_requirements=scenario_dict['user_requirements'],
        num_users=scenario_dict['num_users'],
        initial_uav_position=np.array(scenario_dict['initial_uav_position'])
    )

    # Initialize local SCA solver
    sca_solver = SCASolver(SCAConfig(max_iterations=sca_iterations))

    # Get SCA solution
    sca_pos, _ = sca_solver.solve(scenario, verbose=False)
    sca_metrics = compute_channel_metrics(sca_pos, scenario)
    sca_throughput = sca_metrics['total_throughput']

    # Collect experiences (10 steps)
    experiences = []
    pos = scenario.initial_uav_position.copy()

    for step in range(10):
        # Get physics features
        physics_feats = extract_physics_features(pos, scenario)

        # Get SCA gradient direction
        _, grad = compute_throughput_gradient(pos, scenario)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            sca_direction = grad / grad_norm
        else:
            sca_direction = np.zeros(3)

        # Full state
        state = np.concatenate([physics_feats, sca_direction]).astype(np.float32)

        # Random exploration action (will be replaced with policy action in main process)
        correction = np.random.randn(3) * 0.1 * residual_scale

        # Apply correction to SCA solution
        next_pos = sca_pos + correction
        next_pos[0] = np.clip(next_pos[0], 0, 100)
        next_pos[1] = np.clip(next_pos[1], 0, 100)
        next_pos[2] = np.clip(next_pos[2], 10, 40)

        # Compute reward
        metrics = compute_channel_metrics(next_pos, scenario)
        reward = (metrics['total_throughput'] - sca_throughput) / 10
        reward -= 0.01 * np.linalg.norm(correction)

        # Next state
        next_physics_feats = extract_physics_features(next_pos, scenario)
        _, next_grad = compute_throughput_gradient(next_pos, scenario)
        next_grad_norm = np.linalg.norm(next_grad)
        if next_grad_norm > 1e-6:
            next_sca_direction = next_grad / next_grad_norm
        else:
            next_sca_direction = np.zeros(3)
        next_state = np.concatenate([next_physics_feats, next_sca_direction]).astype(np.float32)

        experiences.append({
            'state': state.tolist(),
            'action': correction.tolist(),
            'reward': float(reward),
            'next_state': next_state.tolist(),
            'done': step == 9,
            'sca_direction': sca_direction.tolist()
        })

        pos = next_pos.copy()

    return experiences


def scenario_to_dict(scenario):
    """Convert Scenario to picklable dict."""
    return {
        'id': scenario.id,
        'bs_position': scenario.bs_position.tolist(),
        'user_positions': [p.tolist() for p in scenario.user_positions],
        'user_requirements': scenario.user_requirements,
        'num_users': scenario.num_users,
        'initial_uav_position': scenario.initial_uav_position.tolist()
    }


def evaluate_baseline(method_name, get_position_fn, scenarios, verbose=False):
    """Evaluate a baseline method on all scenarios."""
    results = []

    for scenario in scenarios:
        start_time = time.time()
        pos = get_position_fn(scenario)
        latency_ms = (time.time() - start_time) * 1000

        metrics = compute_channel_metrics(pos, scenario)
        results.append({
            'scenario_id': scenario.id,
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage': metrics['coverage_rate'],
            'latency_ms': latency_ms
        })

    avg_throughput = np.mean([r['throughput'] for r in results])
    avg_fairness = np.mean([r['fairness'] for r in results])
    avg_coverage = np.mean([r['coverage'] for r in results])
    avg_latency = np.mean([r['latency_ms'] for r in results])

    if verbose:
        print(f"{method_name:20s}: {avg_throughput:7.1f} Mbps, "
              f"fairness={avg_fairness:.3f}, coverage={avg_coverage:.2%}, "
              f"latency={avg_latency:.2f}ms")

    return {
        'method': method_name,
        'avg_throughput': avg_throughput,
        'avg_fairness': avg_fairness,
        'avg_coverage': avg_coverage,
        'avg_latency_ms': avg_latency
    }


def evaluate_sca(scenarios, max_iters=50, verbose=False):
    """Evaluate pure SCA optimization."""
    solver = SCASolver(SCAConfig(max_iterations=max_iters))
    results = []

    for scenario in scenarios:
        start_time = time.time()
        pos, info = solver.solve(scenario, verbose=False)
        latency_ms = (time.time() - start_time) * 1000

        metrics = compute_channel_metrics(pos, scenario)
        results.append({
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage': metrics['coverage_rate'],
            'latency_ms': latency_ms,
            'converged': info['converged']
        })

    avg_throughput = np.mean([r['throughput'] for r in results])
    avg_fairness = np.mean([r['fairness'] for r in results])
    avg_coverage = np.mean([r['coverage'] for r in results])
    avg_latency = np.mean([r['latency_ms'] for r in results])
    convergence_rate = np.mean([r['converged'] for r in results])

    if verbose:
        print(f"{'SCA-' + str(max_iters):20s}: {avg_throughput:7.1f} Mbps, "
              f"fairness={avg_fairness:.3f}, coverage={avg_coverage:.2%}, "
              f"latency={avg_latency:.1f}ms, converged={convergence_rate:.0%}")

    return {
        'method': f'SCA-{max_iters}',
        'avg_throughput': avg_throughput,
        'avg_fairness': avg_fairness,
        'avg_coverage': avg_coverage,
        'avg_latency_ms': avg_latency,
        'convergence_rate': convergence_rate
    }


def evaluate_sgac(agent, scenarios, verbose=False):
    """Evaluate SGAC agent."""
    results = []

    for scenario in scenarios:
        start_time = time.time()
        pos = agent.get_position(scenario, deterministic=True)
        latency_ms = (time.time() - start_time) * 1000

        metrics = compute_channel_metrics(pos, scenario)
        results.append({
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage': metrics['coverage_rate'],
            'latency_ms': latency_ms
        })

    avg_throughput = np.mean([r['throughput'] for r in results])
    avg_fairness = np.mean([r['fairness'] for r in results])
    avg_coverage = np.mean([r['coverage'] for r in results])
    avg_latency = np.mean([r['latency_ms'] for r in results])

    if verbose:
        print(f"{'SGAC (MI-RL)':20s}: {avg_throughput:7.1f} Mbps, "
              f"fairness={avg_fairness:.3f}, coverage={avg_coverage:.2%}, "
              f"latency={avg_latency:.2f}ms")

    return {
        'method': 'SGAC (MI-RL)',
        'avg_throughput': avg_throughput,
        'avg_fairness': avg_fairness,
        'avg_coverage': avg_coverage,
        'avg_latency_ms': avg_latency
    }


def train_sgac_parallel(config, train_scenarios, eval_scenarios,
                        num_episodes=50000, eval_interval=2500,
                        checkpoint_interval=5000, checkpoint_dir=None,
                        resume_from=None, num_workers=4, verbose=True):
    """
    Train SGAC agent with parallel experience collection.

    Args:
        config: SGAC configuration
        train_scenarios: Training scenarios
        eval_scenarios: Evaluation scenarios
        num_episodes: Total episodes to train
        eval_interval: Evaluate every N episodes
        checkpoint_interval: Save checkpoint every N episodes
        checkpoint_dir: Directory for checkpoints
        resume_from: Path to checkpoint to resume from
        num_workers: Number of parallel workers
        verbose: Print progress

    Returns:
        agent: Trained SGAC agent
        training_history: List of evaluation results during training
    """
    agent = SGACAgent(config)
    training_history = []
    start_episode = 1
    best_throughput = 0
    best_episode = 0

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = agent.load_checkpoint(resume_from)
        start_episode = checkpoint.get('episode', 0) + 1
        training_history = checkpoint.get('training_history', [])
        best_throughput = checkpoint.get('best_throughput', 0)
        best_episode = checkpoint.get('best_episode', 0)
        print(f"  Resuming from episode {start_episode}, best={best_throughput:.1f} Mbps")

    print(f"\nTraining SGAC for {num_episodes} episodes (parallel, {num_workers} workers)...")
    print(f"Device: {agent.device}")
    print(f"State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
    print(f"Checkpoint interval: {checkpoint_interval} episodes")
    print("-" * 60)

    # Initial evaluation if starting fresh
    if start_episode == 1:
        eval_result = evaluate_sgac(agent, eval_scenarios, verbose=verbose)
        training_history.append({
            'episode': 0,
            'throughput': eval_result['avg_throughput'],
            'fairness': eval_result['avg_fairness']
        })
        best_throughput = eval_result['avg_throughput']

    # Convert scenarios to dicts for multiprocessing
    train_scenario_dicts = [scenario_to_dict(s) for s in train_scenarios]

    # Training parameters
    sca_iterations = 20
    residual_scale = config.residual_scale
    batch_size = num_workers * 10  # Process multiple episodes per batch

    # Create worker pool
    pool = Pool(processes=num_workers)

    try:
        episode = start_episode
        start_time = time.time()
        last_log_time = start_time

        while episode <= num_episodes:
            # Prepare batch of scenarios
            batch_scenarios = []
            for i in range(batch_size):
                idx = (episode + i) % len(train_scenario_dicts)
                batch_scenarios.append((train_scenario_dicts[idx], sca_iterations, residual_scale))

            # Parallel experience collection
            all_experiences = pool.map(collect_experience_worker, batch_scenarios)

            # Add experiences to replay buffer and train
            for experiences in all_experiences:
                for exp in experiences:
                    # Train on each experience
                    agent.add_experience(
                        np.array(exp['state']),
                        np.array(exp['action']),
                        exp['reward'],
                        np.array(exp['next_state']),
                        exp['done']
                    )

                # Perform gradient update after each episode worth of experiences
                if len(agent.buffer) >= config.batch_size:
                    agent.update()

            episode += batch_size

            # Progress logging every 10 seconds
            current_time = time.time()
            if current_time - last_log_time > 10:
                elapsed = current_time - start_time
                eps_per_sec = (episode - start_episode) / elapsed
                remaining = (num_episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0
                print(f"  Progress: ep {episode}/{num_episodes} "
                      f"({episode/num_episodes*100:.1f}%), "
                      f"{eps_per_sec:.1f} eps/s, "
                      f"ETA: {remaining/60:.1f} min")
                last_log_time = current_time

            # Periodic evaluation
            if episode % eval_interval == 0 or episode >= num_episodes:
                eval_result = evaluate_sgac(agent, eval_scenarios, verbose=verbose)
                training_history.append({
                    'episode': episode,
                    'throughput': eval_result['avg_throughput'],
                    'fairness': eval_result['avg_fairness']
                })

                if eval_result['avg_throughput'] > best_throughput:
                    best_throughput = eval_result['avg_throughput']
                    best_episode = episode

                if verbose:
                    print(f"  Episode {episode}: "
                          f"throughput={eval_result['avg_throughput']:.1f} Mbps "
                          f"(best={best_throughput:.1f} @ ep {best_episode})")

            # Save checkpoint periodically
            if checkpoint_dir and episode % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pt')
                agent.save_checkpoint(checkpoint_path, {
                    'episode': episode,
                    'training_history': training_history,
                    'best_throughput': best_throughput,
                    'best_episode': best_episode
                })
                print(f"  [Checkpoint saved: {checkpoint_path}]")

                # Also save latest checkpoint
                latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
                agent.save_checkpoint(latest_path, {
                    'episode': episode,
                    'training_history': training_history,
                    'best_throughput': best_throughput,
                    'best_episode': best_episode
                })

    finally:
        pool.close()
        pool.join()

    print("-" * 60)
    total_time = time.time() - start_time
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best: {best_throughput:.1f} Mbps @ episode {best_episode}")

    return agent, training_history


def main():
    parser = argparse.ArgumentParser(description='Train Math-Informed RL agent (parallel)')
    parser.add_argument('--episodes', type=int, default=50000,
                        help='Number of training episodes')
    parser.add_argument('--eval-interval', type=int, default=2500,
                        help='Evaluation interval')
    parser.add_argument('--checkpoint-interval', type=int, default=5000,
                        help='Checkpoint save interval')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (or "latest")')
    parser.add_argument('--output-dir', type=str,
                        default='../../results/mi_rl',
                        help='Output directory for results')
    args = parser.parse_args()

    # Set number of workers
    num_workers = args.workers if args.workers else min(cpu_count(), 8)

    print("=" * 70)
    print("Math-Informed RL - Parallel Training")
    print("=" * 70)
    print()
    print(f"Using {num_workers} parallel workers")
    print(f"Available CPUs: {cpu_count()}")
    print()

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Generate scenarios
    print("Generating scenarios...")
    all_scenarios = generate_scenarios(num_scenarios=100, seed=42)
    train_scenarios = all_scenarios[:80]
    eval_scenarios = all_scenarios[80:]

    print(f"  Train scenarios: {len(train_scenarios)}")
    print(f"  Eval scenarios: {len(eval_scenarios)}")
    print()

    # Evaluate baselines first
    print("Evaluating baselines on evaluation set...")
    print("-" * 60)

    baseline_results = {}
    baseline_results['Analytical'] = evaluate_baseline(
        'Analytical', get_position_analytical, eval_scenarios, verbose=True
    )
    baseline_results['SCA-20'] = evaluate_sca(eval_scenarios, max_iters=20, verbose=True)
    baseline_results['SCA-50'] = evaluate_sca(eval_scenarios, max_iters=50, verbose=True)
    print()

    # Configure SGAC
    config = SGACConfig(
        hidden_dim=256,
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99
    )

    # Setup checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Handle resume
    resume_path = None
    if args.resume:
        if args.resume == 'latest':
            resume_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
        else:
            resume_path = args.resume
        if resume_path and not os.path.exists(resume_path):
            print(f"Warning: Checkpoint not found: {resume_path}")
            resume_path = None

    # Train SGAC with parallel workers
    agent, training_history = train_sgac_parallel(
        config, train_scenarios, eval_scenarios,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_path,
        num_workers=num_workers,
        verbose=True
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    sgac_result = evaluate_sgac(agent, eval_scenarios, verbose=True)

    analytical_throughput = baseline_results['Analytical']['avg_throughput']
    sca_throughput = baseline_results['SCA-50']['avg_throughput']
    sgac_throughput = sgac_result['avg_throughput']

    print(f"\n1. SGAC vs Analytical:  {sgac_throughput:.1f} vs {analytical_throughput:.1f} Mbps "
          f"(+{(sgac_throughput/analytical_throughput-1)*100:.1f}%)")
    print(f"2. SGAC vs Pure SCA:    {sgac_throughput:.1f} vs {sca_throughput:.1f} Mbps "
          f"({'+' if sgac_throughput > sca_throughput else ''}"
          f"{(sgac_throughput/sca_throughput-1)*100:.1f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'mi_rl_parallel_results_{timestamp}.json')

    save_data = {
        'config': {
            'episodes': args.episodes,
            'workers': num_workers
        },
        'training_history': training_history,
        'final_results': {
            'SGAC': {
                'avg_throughput': sgac_result['avg_throughput'],
                'avg_fairness': sgac_result['avg_fairness'],
                'avg_coverage': sgac_result['avg_coverage'],
                'avg_latency_ms': sgac_result['avg_latency_ms']
            }
        },
        'timestamp': timestamp
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Save model
    model_file = os.path.join(output_dir, f'sgac_model_{timestamp}.pt')
    agent.save(model_file)
    print(f"Model saved to: {model_file}")


if __name__ == "__main__":
    main()
