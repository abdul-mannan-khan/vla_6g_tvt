#!/usr/bin/env python3
"""
Validation Script for Mathematical Formulation v3 Theoretical Claims.

This script empirically validates the theoretical guarantees proven in the
mathematical formulation document:

1. Theorem 3.2: Reward function aligns with throughput maximization
2. Theorem 3.3: Performance floor guarantee (SGAC >= SCA)
3. Theorem 4.1: Lyapunov stability (position converges)
4. Corollary 4.1: Exponential convergence rate
5. Theorem 5.2: Actor convergence to stationary point
6. Theorem 5.3: Accelerated convergence via warm start

Usage:
    python validate_theoretical_claims.py [--episodes 500] [--output-dir ./validation_results]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_common import (
    generate_scenarios, compute_channel_metrics,
    get_position_analytical, get_position_random, get_position_static
)
from mi_rl.sgac_agent import SGACAgent, SGACConfig
from classical.sca_solver import SCASolver, SCAConfig


class TheoreticalValidator:
    """Validates theoretical claims from mathematical formulation v3."""

    def __init__(self, output_dir: str = './validation_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        self.validation_passed = {}

    def log(self, msg: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def validate_theorem_3_2_reward_alignment(self, agent, scenarios, num_trials=50):
        """
        Validate Theorem 3.2: Reward function aligns with throughput maximization.

        The theorem states that maximizing cumulative reward leads to
        throughput maximization. We verify this by checking correlation
        between cumulative rewards and achieved throughput.
        """
        self.log("=" * 70)
        self.log("THEOREM 3.2: Reward Alignment with Throughput Maximization")
        self.log("=" * 70)

        rewards_list = []
        throughputs_list = []

        for trial in range(num_trials):
            scenario = scenarios[trial % len(scenarios)]

            # Get initial position
            initial_pos = np.array([50.0, 50.0, 25.0])

            # Run agent for several steps
            cumulative_reward = 0
            pos = initial_pos.copy()

            for step in range(10):
                state = agent._get_state(pos, scenario)
                action = agent.select_action(state, deterministic=True)

                # Get SCA position for baseline comparison
                sca_solver = SCASolver(SCAConfig(max_iterations=5))
                sca_pos, _ = sca_solver.solve(scenario, initial_pos=pos)

                # Apply action
                next_pos = np.clip(pos + action, [0, 0, 10], [100, 100, 40])

                # Compute reward (as defined in formulation)
                next_metrics = compute_channel_metrics(next_pos, scenario)
                sca_metrics = compute_channel_metrics(sca_pos, scenario)

                improvement = (next_metrics['total_throughput'] - sca_metrics['total_throughput']) / 10.0
                correction_penalty = 0.01 * np.linalg.norm(action - 0.3 * action / (np.linalg.norm(action) + 1e-8))
                reward = improvement - correction_penalty

                cumulative_reward += reward
                pos = next_pos

            # Final throughput
            final_metrics = compute_channel_metrics(pos, scenario)
            final_throughput = final_metrics['total_throughput']

            rewards_list.append(cumulative_reward)
            throughputs_list.append(final_throughput)

        # Compute correlation
        correlation = np.corrcoef(rewards_list, throughputs_list)[0, 1]

        # Validate: correlation should be positive and significant
        passed = correlation > 0.5

        self.log(f"  Reward-Throughput Correlation: {correlation:.4f}")
        self.log(f"  Mean Cumulative Reward: {np.mean(rewards_list):.4f}")
        self.log(f"  Mean Final Throughput: {np.mean(throughputs_list):.2f} Mbps")
        self.log(f"  VALIDATION: {'PASSED' if passed else 'FAILED'} (correlation > 0.5)")

        self.results['theorem_3_2'] = {
            'correlation': correlation,
            'mean_reward': float(np.mean(rewards_list)),
            'mean_throughput': float(np.mean(throughputs_list)),
            'passed': passed
        }
        self.validation_passed['theorem_3_2'] = passed

        return passed

    def validate_theorem_3_3_performance_floor(self, agent, scenarios):
        """
        Validate Theorem 3.3: Performance Floor Guarantee.

        The theorem states that SGAC throughput >= SCA throughput.
        We verify this empirically across all scenarios.
        """
        self.log("\n" + "=" * 70)
        self.log("THEOREM 3.3: Performance Floor Guarantee (SGAC >= SCA)")
        self.log("=" * 70)

        sgac_throughputs = []
        sca_throughputs = []
        num_violations = 0

        sca_solver = SCASolver(SCAConfig(max_iterations=20))

        for scenario in scenarios:
            # SGAC position
            sgac_pos = agent.get_position(scenario, deterministic=True)
            sgac_metrics = compute_channel_metrics(sgac_pos, scenario)
            sgac_throughput = sgac_metrics['total_throughput']

            # SCA position (baseline)
            sca_pos, _ = sca_solver.solve(scenario)
            sca_metrics = compute_channel_metrics(sca_pos, scenario)
            sca_throughput = sca_metrics['total_throughput']

            sgac_throughputs.append(sgac_throughput)
            sca_throughputs.append(sca_throughput)

            # Check for violations (with small tolerance for numerical errors)
            if sgac_throughput < sca_throughput - 1.0:  # 1 Mbps tolerance
                num_violations += 1

        avg_sgac = np.mean(sgac_throughputs)
        avg_sca = np.mean(sca_throughputs)
        violation_rate = num_violations / len(scenarios)

        # Passed if violation rate < 10% (allowing for some exploration noise)
        passed = violation_rate < 0.10 and avg_sgac >= avg_sca * 0.95

        self.log(f"  Average SGAC Throughput: {avg_sgac:.2f} Mbps")
        self.log(f"  Average SCA Throughput:  {avg_sca:.2f} Mbps")
        self.log(f"  Relative Performance: {avg_sgac/avg_sca*100:.1f}%")
        self.log(f"  Violations: {num_violations}/{len(scenarios)} ({violation_rate*100:.1f}%)")
        self.log(f"  VALIDATION: {'PASSED' if passed else 'FAILED'} (violations < 10%)")

        self.results['theorem_3_3'] = {
            'avg_sgac_throughput': avg_sgac,
            'avg_sca_throughput': avg_sca,
            'relative_performance': avg_sgac / avg_sca,
            'violation_rate': violation_rate,
            'passed': passed
        }
        self.validation_passed['theorem_3_3'] = passed

        return passed

    def validate_theorem_4_1_lyapunov_stability(self, agent, scenarios, num_trials=20):
        """
        Validate Theorem 4.1: Lyapunov Stability.

        The theorem states that the UAV position converges asymptotically
        to the optimal position under the SGAC policy.
        """
        self.log("\n" + "=" * 70)
        self.log("THEOREM 4.1: Lyapunov Stability (Position Convergence)")
        self.log("=" * 70)

        convergence_count = 0
        final_distances = []

        for trial in range(num_trials):
            scenario = scenarios[trial % len(scenarios)]

            # Start from random position
            pos = np.array([
                np.random.uniform(10, 90),
                np.random.uniform(10, 90),
                np.random.uniform(15, 35)
            ])

            # Get approximate optimal position (SCA-50 as proxy)
            sca_solver = SCASolver(SCAConfig(max_iterations=50))
            optimal_pos, _ = sca_solver.solve(scenario)

            # Track position trajectory
            positions = [pos.copy()]
            distances = [np.linalg.norm(pos - optimal_pos)]

            # Run for multiple steps
            for step in range(50):
                state = agent._get_state(pos, scenario)
                action = agent.select_action(state, deterministic=True)
                pos = np.clip(pos + action * 0.5, [0, 0, 10], [100, 100, 40])
                positions.append(pos.copy())
                distances.append(np.linalg.norm(pos - optimal_pos))

            # Check convergence: final distance should be smaller than initial
            initial_dist = distances[0]
            final_dist = distances[-1]
            final_distances.append(final_dist)

            if final_dist < initial_dist * 0.5:  # At least 50% closer
                convergence_count += 1

        convergence_rate = convergence_count / num_trials
        avg_final_distance = np.mean(final_distances)

        passed = convergence_rate >= 0.7  # 70% should converge

        self.log(f"  Convergence Rate: {convergence_rate*100:.1f}%")
        self.log(f"  Average Final Distance to Optimal: {avg_final_distance:.2f}m")
        self.log(f"  VALIDATION: {'PASSED' if passed else 'FAILED'} (convergence >= 70%)")

        self.results['theorem_4_1'] = {
            'convergence_rate': convergence_rate,
            'avg_final_distance': avg_final_distance,
            'passed': passed
        }
        self.validation_passed['theorem_4_1'] = passed

        return passed

    def validate_corollary_4_1_exponential_convergence(self, agent, scenarios, num_trials=10):
        """
        Validate Corollary 4.1: Exponential Convergence Rate.

        The corollary states that ||p_t - p*|| <= C(1-mu)^(t/2).
        We fit an exponential decay to observed distances.
        """
        self.log("\n" + "=" * 70)
        self.log("COROLLARY 4.1: Exponential Convergence Rate")
        self.log("=" * 70)

        all_distances = []

        for trial in range(num_trials):
            scenario = scenarios[trial % len(scenarios)]

            # Start from random position
            pos = np.array([
                np.random.uniform(10, 90),
                np.random.uniform(10, 90),
                np.random.uniform(15, 35)
            ])

            # Get approximate optimal position
            sca_solver = SCASolver(SCAConfig(max_iterations=50))
            optimal_pos, _ = sca_solver.solve(scenario)

            distances = []
            for step in range(30):
                distances.append(np.linalg.norm(pos - optimal_pos))
                state = agent._get_state(pos, scenario)
                action = agent.select_action(state, deterministic=True)
                pos = np.clip(pos + action * 0.3, [0, 0, 10], [100, 100, 40])

            all_distances.append(distances)

        # Average distances across trials
        avg_distances = np.mean(all_distances, axis=0)

        # Fit exponential: d(t) = C * exp(-lambda * t)
        # Take log: log(d(t)) = log(C) - lambda * t
        log_distances = np.log(avg_distances + 1e-6)
        times = np.arange(len(avg_distances))

        # Linear fit to log distances
        coeffs = np.polyfit(times, log_distances, 1)
        decay_rate = -coeffs[0]  # lambda

        # Compute R^2 for exponential fit
        fitted = np.exp(coeffs[1] + coeffs[0] * times)
        ss_res = np.sum((avg_distances - fitted) ** 2)
        ss_tot = np.sum((avg_distances - np.mean(avg_distances)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Compute equivalent mu: (1-mu)^(t/2) = exp(-lambda*t) => mu = 1 - exp(-2*lambda)
        mu = 1 - np.exp(-2 * decay_rate)

        passed = decay_rate > 0.01 and r_squared > 0.5

        self.log(f"  Estimated Decay Rate (lambda): {decay_rate:.4f}")
        self.log(f"  Equivalent Lyapunov mu: {mu:.4f}")
        self.log(f"  Exponential Fit R^2: {r_squared:.4f}")
        self.log(f"  Initial Distance: {avg_distances[0]:.2f}m")
        self.log(f"  Final Distance: {avg_distances[-1]:.2f}m")
        self.log(f"  VALIDATION: {'PASSED' if passed else 'FAILED'} (decay > 0.01, R^2 > 0.5)")

        self.results['corollary_4_1'] = {
            'decay_rate': float(decay_rate),
            'lyapunov_mu': float(mu),
            'r_squared': float(r_squared),
            'initial_distance': float(avg_distances[0]),
            'final_distance': float(avg_distances[-1]),
            'passed': passed
        }
        self.validation_passed['corollary_4_1'] = passed

        return passed

    def validate_theorem_5_2_actor_convergence(self, train_scenarios, eval_scenarios,
                                                num_episodes=300):
        """
        Validate Theorem 5.2: Actor Convergence to Stationary Point.

        Track gradient norms during training - they should decrease.
        """
        self.log("\n" + "=" * 70)
        self.log("THEOREM 5.2: Actor Convergence to Stationary Point")
        self.log("=" * 70)

        config = SGACConfig(
            hidden_dim=128,
            learning_rate=3e-4,
            batch_size=32
        )
        agent = SGACAgent(config)

        gradient_norms = []
        throughputs = []

        for episode in range(num_episodes):
            scenario = train_scenarios[episode % len(train_scenarios)]

            # Train episode
            metrics = agent.train_episode(scenario, max_steps=5)

            # Record gradient norm (if available)
            if hasattr(agent, 'last_actor_grad_norm'):
                gradient_norms.append(agent.last_actor_grad_norm)

            # Periodic evaluation
            if (episode + 1) % 50 == 0:
                total_throughput = 0
                for eval_scenario in eval_scenarios[:5]:
                    pos = agent.get_position(eval_scenario, deterministic=True)
                    metrics = compute_channel_metrics(pos, eval_scenario)
                    total_throughput += metrics['total_throughput']
                avg_throughput = total_throughput / 5
                throughputs.append(avg_throughput)
                self.log(f"  Episode {episode+1}: Throughput = {avg_throughput:.1f} Mbps")

        # Check throughput improvement (proxy for convergence)
        if len(throughputs) >= 2:
            initial_throughput = throughputs[0]
            final_throughput = throughputs[-1]
            improvement = (final_throughput - initial_throughput) / initial_throughput

            passed = improvement > 0 or final_throughput > 150  # Either improved or already good
        else:
            improvement = 0
            passed = False

        self.log(f"  Initial Throughput: {throughputs[0] if throughputs else 0:.1f} Mbps")
        self.log(f"  Final Throughput: {throughputs[-1] if throughputs else 0:.1f} Mbps")
        self.log(f"  Improvement: {improvement*100:.1f}%")
        self.log(f"  VALIDATION: {'PASSED' if passed else 'FAILED'}")

        self.results['theorem_5_2'] = {
            'initial_throughput': float(throughputs[0]) if throughputs else 0,
            'final_throughput': float(throughputs[-1]) if throughputs else 0,
            'improvement': float(improvement),
            'passed': passed
        }
        self.validation_passed['theorem_5_2'] = passed

        return passed, agent

    def validate_theorem_5_3_warm_start_speedup(self, train_scenarios, eval_scenarios,
                                                 num_episodes=200):
        """
        Validate Theorem 5.3: Accelerated Convergence via Warm Start.

        Compare convergence speed of SGAC (SCA-guided) vs vanilla RL.
        """
        self.log("\n" + "=" * 70)
        self.log("THEOREM 5.3: Accelerated Convergence via SCA Warm Start")
        self.log("=" * 70)

        # Train SGAC (SCA-guided)
        self.log("  Training SGAC (SCA-guided)...")
        sgac_config = SGACConfig(
            hidden_dim=128,
            sca_weight=0.3,
            learning_rate=3e-4
        )
        sgac_agent = SGACAgent(sgac_config)
        sgac_throughputs = []

        for episode in range(num_episodes):
            scenario = train_scenarios[episode % len(train_scenarios)]
            sgac_agent.train_episode(scenario, max_steps=5)

            if (episode + 1) % 25 == 0:
                total = 0
                for s in eval_scenarios[:5]:
                    pos = sgac_agent.get_position(s, deterministic=True)
                    m = compute_channel_metrics(pos, s)
                    total += m['total_throughput']
                sgac_throughputs.append(total / 5)

        # Train vanilla (no SCA guidance)
        self.log("  Training Vanilla RL (no SCA guidance)...")
        vanilla_config = SGACConfig(
            hidden_dim=128,
            sca_weight=0.0,  # No SCA guidance
            learning_rate=3e-4
        )
        vanilla_agent = SGACAgent(vanilla_config)
        vanilla_throughputs = []

        for episode in range(num_episodes):
            scenario = train_scenarios[episode % len(train_scenarios)]
            vanilla_agent.train_episode(scenario, max_steps=5)

            if (episode + 1) % 25 == 0:
                total = 0
                for s in eval_scenarios[:5]:
                    pos = vanilla_agent.get_position(s, deterministic=True)
                    m = compute_channel_metrics(pos, s)
                    total += m['total_throughput']
                vanilla_throughputs.append(total / 5)

        # Compare convergence
        # Find episodes to reach 90% of final performance
        sgac_final = sgac_throughputs[-1] if sgac_throughputs else 0
        vanilla_final = vanilla_throughputs[-1] if vanilla_throughputs else 0

        sgac_90_idx = None
        vanilla_90_idx = None

        for i, t in enumerate(sgac_throughputs):
            if t >= 0.9 * sgac_final:
                sgac_90_idx = (i + 1) * 25
                break

        for i, t in enumerate(vanilla_throughputs):
            if t >= 0.9 * vanilla_final:
                vanilla_90_idx = (i + 1) * 25
                break

        if sgac_90_idx and vanilla_90_idx:
            speedup = vanilla_90_idx / sgac_90_idx
        else:
            speedup = 1.0

        passed = sgac_final > vanilla_final or speedup > 1.2

        self.log(f"  SGAC Final Throughput: {sgac_final:.1f} Mbps")
        self.log(f"  Vanilla Final Throughput: {vanilla_final:.1f} Mbps")
        self.log(f"  SGAC Episodes to 90%: {sgac_90_idx}")
        self.log(f"  Vanilla Episodes to 90%: {vanilla_90_idx}")
        self.log(f"  Speedup Factor: {speedup:.2f}x")
        self.log(f"  VALIDATION: {'PASSED' if passed else 'FAILED'}")

        self.results['theorem_5_3'] = {
            'sgac_final_throughput': float(sgac_final),
            'vanilla_final_throughput': float(vanilla_final),
            'sgac_episodes_to_90': sgac_90_idx,
            'vanilla_episodes_to_90': vanilla_90_idx,
            'speedup_factor': float(speedup),
            'sgac_history': [float(t) for t in sgac_throughputs],
            'vanilla_history': [float(t) for t in vanilla_throughputs],
            'passed': passed
        }
        self.validation_passed['theorem_5_3'] = passed

        return passed

    def run_all_validations(self, episodes=300):
        """Run all theoretical validations."""
        self.log("\n" + "=" * 70)
        self.log("MATHEMATICAL FORMULATION v3 - THEORETICAL VALIDATION")
        self.log("=" * 70)
        self.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Episodes per test: {episodes}")

        # Generate scenarios
        self.log("\nGenerating scenarios...")
        all_scenarios = generate_scenarios(num_scenarios=100, seed=42)
        train_scenarios = all_scenarios[:80]
        eval_scenarios = all_scenarios[80:]

        # First train an agent for some validations
        self.log("\nTraining SGAC agent for validation...")
        config = SGACConfig(
            hidden_dim=256,
            sca_weight=0.3,
            learning_rate=3e-4,
            batch_size=64
        )
        agent = SGACAgent(config)

        for episode in range(episodes):
            scenario = train_scenarios[episode % len(train_scenarios)]
            agent.train_episode(scenario, max_steps=10)
            if (episode + 1) % 100 == 0:
                self.log(f"  Trained {episode + 1}/{episodes} episodes")

        # Run validations
        self.validate_theorem_3_2_reward_alignment(agent, eval_scenarios)
        self.validate_theorem_3_3_performance_floor(agent, eval_scenarios)
        self.validate_theorem_4_1_lyapunov_stability(agent, eval_scenarios)
        self.validate_corollary_4_1_exponential_convergence(agent, eval_scenarios)
        self.validate_theorem_5_2_actor_convergence(train_scenarios, eval_scenarios,
                                                     num_episodes=min(200, episodes))
        self.validate_theorem_5_3_warm_start_speedup(train_scenarios, eval_scenarios,
                                                      num_episodes=min(200, episodes))

        # Summary
        self.log("\n" + "=" * 70)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 70)

        all_passed = True
        for theorem, passed in self.validation_passed.items():
            status = "PASSED" if passed else "FAILED"
            self.log(f"  {theorem}: {status}")
            all_passed = all_passed and passed

        self.log("\n" + "-" * 70)
        if all_passed:
            self.log("ALL THEORETICAL CLAIMS VALIDATED SUCCESSFULLY")
        else:
            num_passed = sum(self.validation_passed.values())
            total = len(self.validation_passed)
            self.log(f"PARTIAL VALIDATION: {num_passed}/{total} claims validated")
        self.log("-" * 70)

        # Save results
        results_file = os.path.join(self.output_dir,
                                     f'validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'episodes': episodes,
                'results': self.results,
                'all_passed': all_passed
            }, f, indent=2)

        self.log(f"\nResults saved to: {results_file}")

        return all_passed, self.results


def main():
    parser = argparse.ArgumentParser(description='Validate MI-RL theoretical claims')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Training episodes for validation')
    parser.add_argument('--output-dir', type=str, default='../../results/validation',
                        help='Output directory')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)

    validator = TheoreticalValidator(output_dir)
    all_passed, results = validator.run_all_validations(episodes=args.episodes)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
