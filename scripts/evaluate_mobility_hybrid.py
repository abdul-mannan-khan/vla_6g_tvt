#!/usr/bin/env python3
"""
Hybrid VLA+Analytical Controller — Mobility Evaluation

Between VLA updates (~822ms), the Analytical heuristic tracks users at 100ms.
When a VLA inference completes, its output replaces the Analytical target if
the VLA position would yield higher throughput (>5% margin). This gives the
VLA's superior static quality combined with Analytical's fast tracking.

Runs 25 scenarios at 0/30/60/120 km/h.
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

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DT = 0.1
DURATION = 30.0
UAV_SPEED = 4.0
SPEEDS_KMH = [0, 30, 60, 120]
NUM_SCENARIOS = 25

ANALYTICAL_INTERVAL = DT
OPTIMIZER_INTERVAL = 0.57
HYBRID_VLA_INTERVAL = 0.822  # Default VLA interval
HYBRID_MARGIN = 0.05  # 5% throughput margin to accept VLA position


def load_vla_model():
    if not HAS_TORCH:
        return None, None
    print("Loading merged FP16 VLA model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def vla_inference(model, tokenizer, scenario):
    prompt = format_vla_prompt(scenario)
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(
        full_prompt, return_tensors="pt",
        truncation=True, max_length=384).to(model.device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    latency_s = time.perf_counter() - t0

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    pos, parsed_ok = parse_vla_response(response, scenario)
    return pos, latency_s


def simulate_user_mobility(initial_positions, speed_kmh, duration, dt,
                           bounds=(0, 100), rng=None):
    if rng is None:
        rng = np.random.default_rng(123)

    speed_ms = speed_kmh / 3.6
    num_users = len(initial_positions)
    num_steps = int(duration / dt)

    positions = np.zeros((num_steps, num_users, 3))
    positions[0] = np.array(initial_positions)

    if speed_ms == 0:
        for t in range(1, num_steps):
            positions[t] = positions[0]
        return positions

    angles = rng.uniform(0, 2 * np.pi, size=num_users)
    velocities = np.stack([
        speed_ms * np.cos(angles),
        speed_ms * np.sin(angles),
    ], axis=1)

    lo, hi = bounds
    for t in range(1, num_steps):
        new_xy = positions[t - 1, :, :2] + velocities * dt
        for u in range(num_users):
            for dim in range(2):
                if new_xy[u, dim] < lo:
                    new_xy[u, dim] = 2 * lo - new_xy[u, dim]
                    velocities[u, dim] *= -1
                elif new_xy[u, dim] > hi:
                    new_xy[u, dim] = 2 * hi - new_xy[u, dim]
                    velocities[u, dim] *= -1
                new_xy[u, dim] = np.clip(new_xy[u, dim], lo, hi)
        positions[t, :, :2] = new_xy
        positions[t, :, 2] = 1.0
    return positions


def move_uav_towards(current_pos, target_pos, dt, speed=UAV_SPEED):
    diff = target_pos - current_pos
    dist = np.linalg.norm(diff)
    max_step = speed * dt
    if dist <= max_step:
        return target_pos.copy()
    return current_pos + diff / dist * max_step


# ---------------------------------------------------------------------------
# Simulation with hybrid method
# ---------------------------------------------------------------------------

def run_mobility_sim(scenario, speed_kmh, method, vla_model=None,
                     vla_tokenizer=None, vla_interval=HYBRID_VLA_INTERVAL):
    num_steps = int(DURATION / DT)
    rng = np.random.default_rng(42 + scenario.id)
    user_traj = simulate_user_mobility(
        scenario.user_positions, speed_kmh, DURATION, DT, rng=rng)

    # Decision intervals
    if method == 'analytical':
        decision_interval = ANALYTICAL_INTERVAL
    elif method == 'vla':
        decision_interval = vla_interval
    elif method == 'optimized':
        decision_interval = OPTIMIZER_INTERVAL
    elif method == 'hybrid':
        # Hybrid uses analytical every DT + VLA every vla_interval
        decision_interval = None  # Custom logic below
    else:
        raise ValueError(f"Unknown method: {method}")

    uav_pos = scenario.initial_uav_position.copy()
    target_pos = uav_pos.copy()

    # Hybrid state
    vla_target = None  # Last VLA-suggested position
    analytical_target = None
    time_since_vla = vla_interval  # Trigger first VLA call immediately

    throughputs = []
    fairness_vals = []
    coverage_vals = []
    total_vla_calls = 0

    for t in range(num_steps):
        # Create snapshot
        snap = Scenario(
            id=scenario.id,
            num_users=scenario.num_users,
            user_positions=[user_traj[t, u] for u in range(scenario.num_users)],
            user_requirements=scenario.user_requirements,
            bs_position=scenario.bs_position,
            initial_uav_position=uav_pos.copy(),
        )

        if method == 'hybrid':
            # Always update analytical position (fast tracker)
            analytical_target = get_position_analytical(snap)

            # Check if it's time for a VLA update
            time_since_vla += DT
            if time_since_vla >= vla_interval and vla_model is not None:
                time_since_vla = 0.0
                vla_pos, _ = vla_inference(vla_model, vla_tokenizer, snap)
                total_vla_calls += 1

                # Accept VLA position if it yields >5% throughput gain
                vla_metrics = compute_channel_metrics(vla_pos, snap)
                analytical_metrics = compute_channel_metrics(analytical_target, snap)

                if vla_metrics['total_throughput'] > analytical_metrics['total_throughput'] * (1 + HYBRID_MARGIN):
                    vla_target = vla_pos
                else:
                    vla_target = None  # Don't use this VLA output

            # Use VLA target if available and recent, else analytical
            if vla_target is not None:
                target_pos = vla_target
            else:
                target_pos = analytical_target

        else:
            # Standard methods (non-hybrid)
            if method != 'hybrid':
                time_since_vla += DT
                if time_since_vla >= (decision_interval or DT):
                    time_since_vla = 0.0
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
                            user_requirements=snap.user_requirements)
                        target_pos = result['position']

        uav_pos = move_uav_towards(uav_pos, target_pos, DT)
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


def main():
    import glob

    # Determine VLA interval from latency benchmark results
    vla_interval = HYBRID_VLA_INTERVAL
    latency_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'latency_benchmark_*.json')))
    if latency_files:
        with open(latency_files[-1]) as f:
            lat_data = json.load(f)
        for cfg in lat_data.get('configs', []):
            if cfg['config_name'] == 'merged_fp16_greedy':
                vla_interval = cfg['latency_ms']['mean'] / 1000.0
                print(f"Using VLA interval from benchmark: {vla_interval*1000:.0f}ms")
                break
    print(f"VLA decision interval: {vla_interval*1000:.0f}ms")

    # Load VLA model
    vla_model, vla_tokenizer = None, None
    if HAS_TORCH:
        vla_model, vla_tokenizer = load_vla_model()

    # Generate scenarios (same as original mobility eval)
    all_scenarios = generate_scenarios(100)
    scenarios = all_scenarios[:NUM_SCENARIOS]
    print(f"\nUsing {len(scenarios)} scenarios for mobility evaluation")

    methods = ['analytical', 'hybrid', 'optimized']
    if vla_model is not None:
        methods.insert(1, 'vla')  # analytical, vla, hybrid, optimized

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
                    vla_interval=vla_interval)
                method_results.append(result)

                if (i + 1) % 5 == 0:
                    print(f"    [{i+1}/{len(scenarios)}] "
                          f"tp={result['mean_throughput']:.1f}Mbps "
                          f"fair={result['mean_fairness']:.3f}")

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
            vla_calls = sum(r['total_vla_calls'] for r in method_results)
            print(f"    => {method}: {agg['mean_throughput']:.1f} +/- "
                  f"{agg['ci95_throughput']:.1f} Mbps, "
                  f"fair={agg['mean_fairness']:.3f}, "
                  f"VLA calls={vla_calls}")

        all_results[f'{speed}_kmh'] = speed_results

    # Summary table
    print(f"\n{'='*70}")
    print("HYBRID MOBILITY EVALUATION SUMMARY")
    print(f"{'='*70}")
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
                fair = all_results[key][method]['aggregate']['mean_fairness']
                print(f"  {tp:>6.1f}({fair:.2f})", end='')
            else:
                print(f"  {'N/A':>12s}", end='')
        print()

    # Fairness table
    print(f"\nFairness by speed:")
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
                fair = all_results[key][method]['aggregate']['mean_fairness']
                print(f"  {fair:>12.3f}", end='')
        print()

    # Save
    results_data = {
        'experiment': 'mobility_hybrid',
        'speeds_kmh': SPEEDS_KMH,
        'methods': methods,
        'num_scenarios': NUM_SCENARIOS,
        'duration_s': DURATION,
        'dt_s': DT,
        'vla_interval_s': vla_interval,
        'hybrid_margin': HYBRID_MARGIN,
        'uav_speed_ms': UAV_SPEED,
        'results': all_results,
    }
    save_results(results_data, 'mobility_hybrid')

    if vla_model is not None:
        del vla_model, vla_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == '__main__':
    main()
