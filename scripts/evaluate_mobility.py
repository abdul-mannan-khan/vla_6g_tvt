#!/usr/bin/env python3
"""
M4: Mobility Evaluation

Simulates vehicular movement at 0, 30, 60, 120 km/h with wall-bounce model.
- Timestep: dt=100ms, duration=30s per scenario
- Decision intervals: Analytical=every step, VLA-optimized=based on M1 latency,
  Optimized=every 570ms
- UAV transit: flies to new position at 4 m/s
- 25 scenarios (subset for feasibility)

Methods: analytical, vla (via model), optimized (differential evolution)
"""

import sys
import os
import json
import time
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (
    generate_scenarios, compute_channel_metrics, get_position_analytical,
    format_vla_prompt, parse_vla_response, composite_score,
    save_results, MODEL_DIR, RESULTS_DIR, Scenario,
)
from channel_optimizer import optimize_position

# Try to import VLA model loading
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Simulation parameters
DT = 0.1           # 100 ms timestep
DURATION = 30.0     # 30 seconds per scenario
UAV_SPEED = 4.0     # m/s
SPEEDS_KMH = [0, 30, 60, 120]
NUM_SCENARIOS = 25  # Subset for feasibility

# Decision intervals (seconds)
ANALYTICAL_INTERVAL = DT              # Every timestep
OPTIMIZER_INTERVAL = 0.57             # 570 ms (DE solve time)
# VLA interval will be set from M1 results or default to ~0.5s


def load_vla_model():
    """Load the merged FP16 VLA model (fast config from M1)."""
    if not HAS_TORCH:
        return None, None

    print("Loading merged FP16 VLA model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def vla_inference(model, tokenizer, scenario):
    """Run VLA inference on a scenario, return position + latency."""
    prompt = format_vla_prompt(scenario)
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(
        full_prompt, return_tensors="pt",
        truncation=True, max_length=384,
    ).to(model.device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    latency_s = time.perf_counter() - t0

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    pos, parsed_ok = parse_vla_response(response, scenario)
    return pos, latency_s


# ---------------------------------------------------------------------------
# Mobility model: constant velocity, random direction, wall-bounce
# ---------------------------------------------------------------------------

def simulate_user_mobility(initial_positions, speed_kmh, duration, dt,
                           bounds=(0, 100), rng=None):
    """Simulate user movement with wall-bounce at boundaries.

    Returns: positions array of shape (num_steps, num_users, 3)
    """
    if rng is None:
        rng = np.random.default_rng(123)

    speed_ms = speed_kmh / 3.6  # Convert km/h to m/s
    num_users = len(initial_positions)
    num_steps = int(duration / dt)

    positions = np.zeros((num_steps, num_users, 3))
    positions[0] = np.array(initial_positions)

    if speed_ms == 0:
        # Static users
        for t in range(1, num_steps):
            positions[t] = positions[0]
        return positions

    # Random initial direction per user (2D movement only, z stays at 1.0)
    angles = rng.uniform(0, 2 * np.pi, size=num_users)
    velocities = np.stack([
        speed_ms * np.cos(angles),
        speed_ms * np.sin(angles),
    ], axis=1)  # (num_users, 2)

    lo, hi = bounds

    for t in range(1, num_steps):
        new_xy = positions[t - 1, :, :2] + velocities * dt

        # Wall-bounce
        for u in range(num_users):
            for dim in range(2):
                if new_xy[u, dim] < lo:
                    new_xy[u, dim] = 2 * lo - new_xy[u, dim]
                    velocities[u, dim] *= -1
                elif new_xy[u, dim] > hi:
                    new_xy[u, dim] = 2 * hi - new_xy[u, dim]
                    velocities[u, dim] *= -1
                # Clamp to avoid numerical drift
                new_xy[u, dim] = np.clip(new_xy[u, dim], lo, hi)

        positions[t, :, :2] = new_xy
        positions[t, :, 2] = 1.0  # Ground users stay at z=1.0

    return positions


# ---------------------------------------------------------------------------
# UAV transition model
# ---------------------------------------------------------------------------

def move_uav_towards(current_pos, target_pos, dt, speed=UAV_SPEED):
    """Move UAV towards target at given speed. Returns new position."""
    diff = target_pos - current_pos
    dist = np.linalg.norm(diff)
    max_step = speed * dt
    if dist <= max_step:
        return target_pos.copy()
    return current_pos + diff / dist * max_step


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_mobility_sim(scenario, speed_kmh, method, vla_model=None,
                     vla_tokenizer=None, vla_interval=0.5):
    """Run mobility simulation for one scenario, one speed, one method.

    Returns dict with time-averaged throughput, fairness, coverage.
    """
    num_steps = int(DURATION / DT)

    # Simulate user movement
    rng = np.random.default_rng(42 + scenario.id)
    user_traj = simulate_user_mobility(
        scenario.user_positions, speed_kmh, DURATION, DT, rng=rng)

    # Determine decision interval
    if method == 'analytical':
        decision_interval = ANALYTICAL_INTERVAL
    elif method == 'vla':
        decision_interval = vla_interval
    elif method == 'optimized':
        decision_interval = OPTIMIZER_INTERVAL
    else:
        raise ValueError(f"Unknown method: {method}")

    # UAV starts at initial position
    uav_pos = scenario.initial_uav_position.copy()
    target_pos = uav_pos.copy()

    throughputs = []
    fairness_vals = []
    coverage_vals = []

    time_since_decision = decision_interval  # Trigger first decision immediately
    total_vla_calls = 0

    for t in range(num_steps):
        current_time = t * DT

        # Create snapshot scenario with current user positions
        snap = Scenario(
            id=scenario.id,
            num_users=scenario.num_users,
            user_positions=[user_traj[t, u] for u in range(scenario.num_users)],
            user_requirements=scenario.user_requirements,
            bs_position=scenario.bs_position,
            initial_uav_position=uav_pos.copy(),
        )

        # Check if it's time for a new decision
        time_since_decision += DT
        if time_since_decision >= decision_interval:
            time_since_decision = 0.0

            if method == 'analytical':
                target_pos = get_position_analytical(snap)
            elif method == 'vla':
                if vla_model is not None:
                    target_pos, _ = vla_inference(vla_model, vla_tokenizer, snap)
                    total_vla_calls += 1
                else:
                    target_pos = get_position_analytical(snap)
            elif method == 'optimized':
                result = optimize_position(
                    bs_position=snap.bs_position,
                    user_positions=snap.user_positions,
                    user_requirements=snap.user_requirements,
                )
                target_pos = result['position']

        # Move UAV towards target
        uav_pos = move_uav_towards(uav_pos, target_pos, DT)

        # Compute metrics at current UAV position
        metrics = compute_channel_metrics(uav_pos, snap)
        throughputs.append(metrics['total_throughput'])
        fairness_vals.append(metrics['fairness'])
        coverage_vals.append(metrics['coverage_rate'])

    return {
        'mean_throughput': float(np.mean(throughputs)),
        'mean_fairness': float(np.mean(fairness_vals)),
        'mean_coverage': float(np.mean(coverage_vals)),
        'std_throughput': float(np.std(throughputs)),
        'total_vla_calls': total_vla_calls,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Determine VLA interval from M1 results if available
    vla_interval = 0.5  # Default: 500ms
    import glob
    latency_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'latency_benchmark_*.json')))
    if latency_files:
        with open(latency_files[-1]) as f:
            lat_data = json.load(f)
        # Use the merged_fp16_greedy config if available
        for cfg in lat_data.get('configs', []):
            if cfg['config_name'] == 'merged_fp16_greedy':
                vla_interval = cfg['latency_ms']['mean'] / 1000.0
                print(f"Using VLA interval from M1: {vla_interval*1000:.0f}ms")
                break
    else:
        print(f"No M1 results found, using default VLA interval: {vla_interval*1000:.0f}ms")

    # Load VLA model
    vla_model, vla_tokenizer = None, None
    if HAS_TORCH:
        vla_model, vla_tokenizer = load_vla_model()

    # Generate evaluation scenarios (first 25 of the canonical 100)
    all_scenarios = generate_scenarios(100)
    scenarios = all_scenarios[:NUM_SCENARIOS]
    print(f"\nUsing {len(scenarios)} scenarios for mobility evaluation")

    methods = ['analytical', 'optimized']
    if vla_model is not None:
        methods.append('vla')

    all_results = {}

    for speed in SPEEDS_KMH:
        print(f"\n{'='*60}")
        print(f"Speed: {speed} km/h")
        print(f"{'='*60}")

        speed_results = {}

        for method in methods:
            print(f"\n  Method: {method}")
            method_results = []

            for i, scenario in enumerate(scenarios):
                result = run_mobility_sim(
                    scenario, speed, method,
                    vla_model=vla_model,
                    vla_tokenizer=vla_tokenizer,
                    vla_interval=vla_interval,
                )
                method_results.append(result)

                if (i + 1) % 5 == 0:
                    print(f"    [{i+1}/{len(scenarios)}] "
                          f"tp={result['mean_throughput']:.1f}Mbps")

            tp_vals = [r['mean_throughput'] for r in method_results]
            fair_vals = [r['mean_fairness'] for r in method_results]
            cov_vals = [r['mean_coverage'] for r in method_results]

            speed_results[method] = {
                'per_scenario': method_results,
                'aggregate': {
                    'mean_throughput': float(np.mean(tp_vals)),
                    'std_throughput': float(np.std(tp_vals)),
                    'ci95_throughput': float(
                        1.96 * np.std(tp_vals, ddof=1) / np.sqrt(len(tp_vals))
                    ),
                    'mean_fairness': float(np.mean(fair_vals)),
                    'mean_coverage': float(np.mean(cov_vals)),
                },
            }

            agg = speed_results[method]['aggregate']
            print(f"    => {method}: {agg['mean_throughput']:.1f} +/- "
                  f"{agg['ci95_throughput']:.1f} Mbps, "
                  f"fair={agg['mean_fairness']:.3f}, "
                  f"cov={agg['mean_coverage']*100:.1f}%")

        all_results[f'{speed}_kmh'] = speed_results

    # Summary table
    print(f"\n{'='*60}")
    print("MOBILITY EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Speed':>8s}", end='')
    for method in methods:
        print(f"  {method:>12s}", end='')
    print()
    print("-" * (8 + 14 * len(methods)))
    for speed in SPEEDS_KMH:
        key = f'{speed}_kmh'
        print(f"{speed:>5d} km/h", end='')
        for method in methods:
            if method in all_results[key]:
                tp = all_results[key][method]['aggregate']['mean_throughput']
                print(f"  {tp:>10.1f}  ", end='')
            else:
                print(f"  {'N/A':>10s}  ", end='')
        print()

    results = {
        'speeds_kmh': SPEEDS_KMH,
        'methods': methods,
        'num_scenarios': NUM_SCENARIOS,
        'duration_s': DURATION,
        'dt_s': DT,
        'vla_interval_s': vla_interval,
        'uav_speed_ms': UAV_SPEED,
        'results': all_results,
    }

    path = save_results(results, 'mobility_evaluation')

    # Clean up
    if vla_model is not None:
        del vla_model, vla_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone! Results at {path}")


if __name__ == '__main__':
    main()
