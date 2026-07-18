#!/usr/bin/env python3
"""
Evaluation Script for VLA-6G UAV Relay System

Compares:
1. VLA-based positioning (fine-tuned model)
2. Analytical baseline
3. Random positioning
4. Static center positioning

Metrics:
- Total throughput
- Fairness index
- Coverage rate
- Trajectory efficiency
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

import numpy as np
import json
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict, field
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from scipy import stats

# Add scripts dir for channel_optimizer import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel_optimizer import optimize_position


@dataclass
class EvaluationResult:
    """Results for a single evaluation run."""
    method: str
    scenario_id: int

    # Communication metrics
    total_throughput: float
    average_user_rate: float
    min_user_rate: float
    fairness_index: float
    coverage_rate: float  # Fraction of users meeting QoS

    # Navigation metrics
    trajectory_length: float
    repositioning_time: float

    # Efficiency
    throughput_per_meter: float  # Throughput gain / trajectory length

    # Timing
    inference_time_ms: float = 0.0


@dataclass
class Scenario:
    """Test scenario configuration."""
    id: int
    num_users: int
    user_positions: List[np.ndarray]
    user_requirements: List[float]  # Required rate per user
    bs_position: np.ndarray
    initial_uav_position: np.ndarray


class EvaluationNode(Node):
    """Node for evaluating different positioning methods."""

    def __init__(self):
        super().__init__('evaluation_node')

        # Parameters
        self.declare_parameter('num_scenarios', 100)
        self.declare_parameter('output_dir', '/home/it-services/ros2_ws/src/vla_6g_tvt/results')
        self.declare_parameter('methods', ['vla', 'analytical', 'random', 'static', 'optimized'])

        self.num_scenarios = self.get_parameter('num_scenarios').value
        self.output_dir = self.get_parameter('output_dir').value
        self.methods = self.get_parameter('methods').value

        os.makedirs(self.output_dir, exist_ok=True)

        # Channel parameters
        self.frequency_ghz = 300.0
        self.bandwidth_ghz = 10.0
        self.bs_power_dbm = 30.0
        self.uav_power_dbm = 20.0
        self.required_rate_default = 25.0  # Mbps (realistic for 300 GHz at typical relay distances)

        # State
        self.results: List[EvaluationResult] = []
        self.current_scenario = None
        self.bs_position = np.array([0.0, 0.0, 30.0])

        # Parse failure tracking
        self._parse_failures = 0
        self._parse_attempts = 0

        # Load Llama model for VLA evaluation
        self.llama_model = None
        self.llama_tokenizer = None
        if 'vla' in self.methods:
            self._load_llama_model()

        # Generate test scenarios
        self.scenarios = self._generate_scenarios()

        self.get_logger().info(f'Evaluation initialized with {len(self.scenarios)} scenarios')
        self.get_logger().info(f'Methods: {self.methods}')

        # Run evaluation
        self.run_evaluation()

    def _generate_scenarios(self) -> List[Scenario]:
        """Generate diverse test scenarios."""
        scenarios = []
        random.seed(42)  # Reproducibility

        for i in range(self.num_scenarios):
            # Vary number of users
            num_users = random.choice([3, 4, 5, 6, 7])

            # Generate user positions with different distributions
            if i % 4 == 0:
                # Clustered users
                center = np.array([random.uniform(40, 60), random.uniform(40, 60)])
                user_positions = [
                    np.array([
                        center[0] + random.gauss(0, 5),
                        center[1] + random.gauss(0, 5),
                        1.0
                    ])
                    for _ in range(num_users)
                ]
            elif i % 4 == 1:
                # Spread users
                user_positions = [
                    np.array([
                        random.uniform(20, 80),
                        random.uniform(20, 80),
                        1.0
                    ])
                    for _ in range(num_users)
                ]
            elif i % 4 == 2:
                # Line configuration
                y_pos = random.uniform(30, 70)
                user_positions = [
                    np.array([20 + i * 60 / (num_users - 1), y_pos, 1.0])
                    for i in range(num_users)
                ]
            else:
                # Circle configuration
                center = np.array([50, 50])
                radius = random.uniform(15, 30)
                user_positions = [
                    np.array([
                        center[0] + radius * np.cos(2 * np.pi * i / num_users),
                        center[1] + radius * np.sin(2 * np.pi * i / num_users),
                        1.0
                    ])
                    for i in range(num_users)
                ]

            # User requirements (varied)
            user_requirements = [
                random.uniform(10, 50) for _ in range(num_users)
            ]

            # Random initial UAV position
            initial_uav = np.array([
                random.uniform(30, 70),
                random.uniform(30, 70),
                random.uniform(15, 35)
            ])

            scenarios.append(Scenario(
                id=i,
                num_users=num_users,
                user_positions=user_positions,
                user_requirements=user_requirements,
                bs_position=self.bs_position,
                initial_uav_position=initial_uav
            ))

        return scenarios

    def compute_channel_metrics(self, uav_pos: np.ndarray,
                                 scenario: Scenario) -> Dict:
        """Compute channel metrics for given UAV position."""
        def calc_snr(d, power):
            d_km = max(d / 1000.0, 0.001)
            fspl = 20 * np.log10(d_km) + 20 * np.log10(self.frequency_ghz) + 92.45
            absorption = 10.0 * d_km
            path_loss = fspl + absorption
            noise = -174 + 10 * np.log10(self.bandwidth_ghz * 1e9) + 10
            return power - path_loss - noise

        d_bs_uav = np.linalg.norm(uav_pos - scenario.bs_position)
        snr_bs_uav = calc_snr(d_bs_uav, self.bs_power_dbm)

        user_rates = []
        users_covered = 0

        for i, user_pos in enumerate(scenario.user_positions):
            d_uav_user = np.linalg.norm(uav_pos - user_pos)
            snr_uav_user = calc_snr(d_uav_user, self.uav_power_dbm)
            effective_snr = min(snr_bs_uav, snr_uav_user)

            snr_linear = 10 ** (effective_snr / 10)
            rate = self.bandwidth_ghz * 1000 * np.log2(1 + max(snr_linear, 0.001))
            rate = min(rate, 10000)

            user_rates.append(rate)

            if rate >= scenario.user_requirements[i]:
                users_covered += 1

        total_throughput = sum(user_rates)

        # Jain's fairness
        if user_rates:
            s = sum(user_rates)
            s2 = sum(r ** 2 for r in user_rates)
            n = len(user_rates)
            fairness = (s ** 2) / (n * s2) if s2 > 0 else 0
        else:
            fairness = 0

        return {
            'total_throughput': total_throughput,
            'average_rate': np.mean(user_rates),
            'min_rate': min(user_rates),
            'fairness': fairness,
            'coverage_rate': users_covered / len(user_rates),
            'user_rates': user_rates
        }

    def get_position_analytical(self, scenario: Scenario) -> np.ndarray:
        """Analytical optimal position."""
        user_centroid = np.mean(scenario.user_positions, axis=0)
        bs_2d = scenario.bs_position[:2]
        centroid_2d = user_centroid[:2]

        alpha = 0.6
        optimal_xy = (1 - alpha) * bs_2d + alpha * centroid_2d

        distances = [np.linalg.norm(optimal_xy - u[:2]) for u in scenario.user_positions]
        avg_dist = np.mean(distances)
        optimal_z = np.clip(0.5 * avg_dist, 10.0, 40.0)

        return np.array([optimal_xy[0], optimal_xy[1], optimal_z])

    def get_position_random(self, scenario: Scenario) -> np.ndarray:
        """Random position within operational area."""
        return np.array([
            random.uniform(20, 80),
            random.uniform(20, 80),
            random.uniform(10, 40)
        ])

    def get_position_static(self, scenario: Scenario) -> np.ndarray:
        """Static center position."""
        return np.array([50.0, 50.0, 25.0])

    def _load_llama_model(self):
        """Load fine-tuned TinyLlama with LoRA adapters for VLA inference."""
        model_path = '/home/it-services/ros2_ws/src/vla_6g_tvt/models/vla_6g_v1'
        base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        self.get_logger().info(f"Loading Llama model from {model_path}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.llama_model = PeftModel.from_pretrained(self.llama_model, model_path)
        self.llama_model.eval()

        self.get_logger().info(f"Llama model loaded on {self.llama_model.device}")

    def _format_prompt(self, scenario: Scenario) -> str:
        """Build prompt from Scenario data matching the training format."""
        prompt = f"""You are a UAV relay positioning expert for 6G networks.

Current situation:
- Base station at: ({scenario.bs_position[0]:.1f}, {scenario.bs_position[1]:.1f}, {scenario.bs_position[2]:.1f})
- UAV currently at: ({scenario.initial_uav_position[0]:.1f}, {scenario.initial_uav_position[1]:.1f}, {scenario.initial_uav_position[2]:.1f})
- Number of ground users: {scenario.num_users}
- Current total throughput: 0.0 Mbps
- Current fairness index: 0.000

User details:
"""
        for i, user_pos in enumerate(scenario.user_positions):
            req = scenario.user_requirements[i]
            prompt += f"  User {i}: pos=({user_pos[0]:.1f}, {user_pos[1]:.1f}), "
            prompt += f"SNR=0.0dB, rate=0.0Mbps, "
            prompt += f"required={req:.1f}Mbps, covered=False\n"

        prompt += """
Task: Determine the optimal UAV relay position.
Output ONLY valid JSON: {"x": float, "y": float, "z": float}
"""
        return prompt

    def _parse_response(self, response: str, scenario: Scenario) -> Tuple[np.ndarray, bool]:
        """Extract x, y, z from model output. Returns (position, parsed_ok)."""
        try:
            # 1) json.loads on first JSON object
            json_match = re.search(r'\{[^{}]+\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    if 'x' in data and 'y' in data and 'z' in data:
                        x, y, z = float(data['x']), float(data['y']), float(data['z'])
                        return np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)]), True
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

            # 2) Regex: "x": val, "y": val, "z": val
            simple_pattern = r'"x":\s*([\d.]+)\s*,\s*"y":\s*([\d.]+)\s*,\s*"z":\s*([\d.]+)'
            match = re.search(simple_pattern, response, re.IGNORECASE)
            if match:
                x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                return np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)]), True

            # 3) Tuple pattern
            tuple_pattern = r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)'
            match = re.search(tuple_pattern, response)
            if match:
                x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                return np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)]), True

            # 4) Number-triplet fallback
            numbers = re.findall(r'(\d+\.?\d*)', response)
            if len(numbers) >= 3:
                x, y, z = float(numbers[0]), float(numbers[1]), float(numbers[2])
                if 0 <= x <= 100 and 0 <= y <= 100 and 1 <= z <= 60:
                    return np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)]), True
        except Exception:
            pass

        # Fallback to analytical
        self.get_logger().warn("Llama parse failed, falling back to analytical")
        return self.get_position_analytical(scenario), False

    def get_position_vla(self, scenario: Scenario) -> np.ndarray:
        """VLA model position using real Llama inference."""
        if self.llama_model is None:
            self.get_logger().warn("Llama model not loaded, falling back to analytical")
            return self.get_position_analytical(scenario)

        prompt = self._format_prompt(scenario)
        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        inputs = self.llama_tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=384
        ).to(self.llama_model.device)

        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.llama_tokenizer.pad_token_id,
            )

        response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        pos, parsed_ok = self._parse_response(response, scenario)
        if not parsed_ok:
            self._parse_failures += 1
        self._parse_attempts += 1
        return pos

    def get_position_optimized(self, scenario: Scenario) -> np.ndarray:
        """Optimization-based position using scipy differential evolution."""
        result = optimize_position(
            bs_position=scenario.bs_position,
            user_positions=scenario.user_positions,
            user_requirements=scenario.user_requirements,
        )
        return result['position']

    def evaluate_method(self, method: str, scenario: Scenario) -> EvaluationResult:
        """Evaluate a single method on a scenario."""
        # Get target position based on method, measuring time
        t0 = time.time()
        if method == 'analytical':
            target_pos = self.get_position_analytical(scenario)
        elif method == 'random':
            target_pos = self.get_position_random(scenario)
        elif method == 'static':
            target_pos = self.get_position_static(scenario)
        elif method == 'vla':
            target_pos = self.get_position_vla(scenario)
        elif method == 'optimized':
            target_pos = self.get_position_optimized(scenario)
        else:
            raise ValueError(f"Unknown method: {method}")
        inference_time_ms = (time.time() - t0) * 1000

        # Compute metrics at target position
        metrics = self.compute_channel_metrics(target_pos, scenario)

        # Trajectory length from initial position
        traj_length = np.linalg.norm(target_pos - scenario.initial_uav_position)

        # Simulated repositioning time (assuming 4 m/s average speed)
        reposition_time = traj_length / 4.0

        # Compute initial metrics for comparison
        initial_metrics = self.compute_channel_metrics(
            scenario.initial_uav_position, scenario
        )

        # Throughput gain per meter traveled
        throughput_gain = metrics['total_throughput'] - initial_metrics['total_throughput']
        throughput_per_meter = throughput_gain / max(traj_length, 0.1)

        return EvaluationResult(
            method=method,
            scenario_id=scenario.id,
            total_throughput=metrics['total_throughput'],
            average_user_rate=metrics['average_rate'],
            min_user_rate=metrics['min_rate'],
            fairness_index=metrics['fairness'],
            coverage_rate=metrics['coverage_rate'],
            trajectory_length=traj_length,
            repositioning_time=reposition_time,
            throughput_per_meter=throughput_per_meter,
            inference_time_ms=inference_time_ms,
        )

    def run_evaluation(self):
        """Run full evaluation."""
        self.get_logger().info("Starting evaluation...")

        for scenario in self.scenarios:
            for method in self.methods:
                result = self.evaluate_method(method, scenario)
                self.results.append(result)

            if scenario.id % 5 == 0:
                self.get_logger().info(f"Completed scenario {scenario.id}/{len(self.scenarios)}")

        # Analyze and save results
        self.analyze_results()
        self.save_results()

    def _ci(self, values):
        """Compute mean and 95% CI half-width."""
        arr = np.array(values)
        mean = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
        return mean, 1.96 * se

    def analyze_results(self):
        """Analyze evaluation results with statistical rigor."""
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("EVALUATION RESULTS SUMMARY")
        self.get_logger().info("=" * 60)

        # Parse failure rate
        if self._parse_attempts > 0:
            fail_rate = self._parse_failures / self._parse_attempts * 100
            self.get_logger().info(f"\nVLA Parse Failure Rate: {fail_rate:.1f}% ({self._parse_failures}/{self._parse_attempts})")

        for method in self.methods:
            method_results = [r for r in self.results if r.method == method]

            tp_mean, tp_ci = self._ci([r.total_throughput for r in method_results])
            fair_mean, fair_ci = self._ci([r.fairness_index for r in method_results])
            cov_mean, cov_ci = self._ci([r.coverage_rate for r in method_results])
            time_mean, time_ci = self._ci([r.inference_time_ms for r in method_results])

            self.get_logger().info(f"\n{method.upper()}:")
            self.get_logger().info(f"  Throughput: {tp_mean:.1f} +/- {tp_ci:.1f} Mbps")
            self.get_logger().info(f"  Fairness:   {fair_mean:.3f} +/- {fair_ci:.3f}")
            self.get_logger().info(f"  Coverage:   {cov_mean * 100:.1f} +/- {cov_ci * 100:.1f}%")
            self.get_logger().info(f"  Latency:    {time_mean:.1f} +/- {time_ci:.1f} ms")

        # Paired t-tests: VLA vs analytical, VLA vs optimized
        self.get_logger().info("\nSTATISTICAL TESTS (paired t-test on throughput):")
        self.get_logger().info("-" * 60)
        vla_results = sorted([r for r in self.results if r.method == 'vla'], key=lambda r: r.scenario_id)
        for compare_method in ['analytical', 'optimized']:
            other_results = sorted([r for r in self.results if r.method == compare_method], key=lambda r: r.scenario_id)
            if len(vla_results) == len(other_results) and len(vla_results) > 1:
                vla_tp = [r.total_throughput for r in vla_results]
                other_tp = [r.total_throughput for r in other_results]
                t_stat, p_value = stats.ttest_rel(vla_tp, other_tp)
                sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
                self.get_logger().info(f"  VLA vs {compare_method}: t={t_stat:.2f}, p={p_value:.4f} {sig}")

        # Per-scenario-type breakdown
        scenario_types = {0: 'clustered', 1: 'spread', 2: 'line', 3: 'circle'}
        self.get_logger().info("\nPER-SCENARIO-TYPE BREAKDOWN:")
        self.get_logger().info("-" * 60)
        for type_id, type_name in scenario_types.items():
            scenario_ids = [s.id for s in self.scenarios if s.id % 4 == type_id]
            self.get_logger().info(f"\n  {type_name.upper()} ({len(scenario_ids)} scenarios):")
            for method in self.methods:
                type_results = [r for r in self.results
                                if r.method == method and r.scenario_id in scenario_ids]
                if type_results:
                    avg_tp = np.mean([r.total_throughput for r in type_results])
                    avg_fair = np.mean([r.fairness_index for r in type_results])
                    avg_cov = np.mean([r.coverage_rate for r in type_results])
                    self.get_logger().info(
                        f"    {method:12s}: throughput={avg_tp:8.1f}  "
                        f"fairness={avg_fair:.3f}  coverage={avg_cov*100:.1f}%"
                    )

        self.get_logger().info("=" * 60)

    def save_results(self):
        """Save results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f'evaluation_{timestamp}.json')

        # Convert results to dict
        results_dict = [asdict(r) for r in self.results]

        # Compute summary statistics
        summary = {}
        for method in self.methods:
            method_results = [r for r in self.results if r.method == method]
            tp_vals = [r.total_throughput for r in method_results]
            tp_mean, tp_ci = self._ci(tp_vals)
            summary[method] = {
                'avg_throughput': float(tp_mean),
                'ci_throughput': float(tp_ci),
                'std_throughput': float(np.std(tp_vals)),
                'avg_fairness': float(np.mean([r.fairness_index for r in method_results])),
                'avg_coverage': float(np.mean([r.coverage_rate for r in method_results])),
                'avg_trajectory_length': float(np.mean([r.trajectory_length for r in method_results])),
                'avg_inference_time_ms': float(np.mean([r.inference_time_ms for r in method_results])),
            }

        with open(filename, 'w') as f:
            json.dump({
                'metadata': {
                    'num_scenarios': self.num_scenarios,
                    'methods': self.methods,
                    'timestamp': timestamp
                },
                'summary': summary,
                'results': results_dict
            }, f, indent=2)

        self.get_logger().info(f"Results saved to {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
