#!/usr/bin/env python3
"""
M1: Latency Benchmark for VLA Inference

Benchmarks 4 configurations on 100 scenarios:
  1. Baseline:   4-bit quantized, temp=0.3, max_tokens=150, do_sample=True
  2. Reduced:    4-bit quantized, temp=0.3, max_tokens=50, do_sample=True
  3. Greedy:     4-bit quantized, max_tokens=50, do_sample=False
  4. Merged FP16: LoRA merged, fp16, max_tokens=50, do_sample=False

5 warmup runs, then 100 scenarios per config.
Reports mean/std/p50/p95 latency and throughput.
"""

import sys
import os
import time
import json
import gc
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (
    generate_scenarios, compute_channel_metrics,
    format_vla_prompt, parse_vla_response,
    get_position_analytical, save_results, MODEL_DIR,
)

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_WARMUP = 5
NUM_SCENARIOS = 100


def load_model_4bit():
    """Load the 4-bit quantized model with LoRA adapters (baseline)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    model.eval()
    return model, tokenizer


def load_model_merged_fp16():
    """Merge LoRA adapters and load as FP16 (key optimization)."""
    print("Loading base model for merging...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in fp16 (no quantization) for merging
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    print("Merging LoRA adapters...")
    model = model.merge_and_unload()
    model.eval()
    print(f"Merged model on {next(model.parameters()).device}, "
          f"dtype={next(model.parameters()).dtype}")
    return model, tokenizer


def run_inference(model, tokenizer, scenario, max_new_tokens=150,
                  temperature=0.3, do_sample=True):
    """Run a single VLA inference. Returns (position, parsed_ok, latency_ms)."""
    prompt = format_vla_prompt(scenario)
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    ).to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs['temperature'] = temperature
        gen_kwargs['do_sample'] = True
    else:
        gen_kwargs['do_sample'] = False

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    pos, parsed_ok = parse_vla_response(response, scenario)
    return pos, parsed_ok, latency_ms


def benchmark_config(model, tokenizer, scenarios, config_name,
                     max_new_tokens, temperature, do_sample):
    """Benchmark one configuration on all scenarios."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"  max_new_tokens={max_new_tokens}, temp={temperature}, "
          f"do_sample={do_sample}")
    print(f"{'='*60}")

    # Warmup
    print(f"  Warmup ({NUM_WARMUP} runs)...")
    for i in range(NUM_WARMUP):
        run_inference(model, tokenizer, scenarios[i],
                      max_new_tokens, temperature, do_sample)

    # Benchmark
    latencies = []
    throughputs = []
    parse_failures = 0
    fairness_vals = []
    coverage_vals = []

    for i, scenario in enumerate(scenarios):
        pos, parsed_ok, lat = run_inference(
            model, tokenizer, scenario,
            max_new_tokens, temperature, do_sample,
        )
        latencies.append(lat)

        if not parsed_ok:
            parse_failures += 1

        metrics = compute_channel_metrics(pos, scenario)
        throughputs.append(metrics['total_throughput'])
        fairness_vals.append(metrics['fairness'])
        coverage_vals.append(metrics['coverage_rate'])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(scenarios)}] lat={lat:.0f}ms, "
                  f"tp={metrics['total_throughput']:.1f}Mbps")

    lat_arr = np.array(latencies)
    tp_arr = np.array(throughputs)

    result = {
        'config_name': config_name,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'do_sample': do_sample,
        'num_scenarios': len(scenarios),
        'parse_failures': parse_failures,
        'latency_ms': {
            'mean': float(np.mean(lat_arr)),
            'std': float(np.std(lat_arr)),
            'p50': float(np.percentile(lat_arr, 50)),
            'p95': float(np.percentile(lat_arr, 95)),
            'min': float(np.min(lat_arr)),
            'max': float(np.max(lat_arr)),
        },
        'throughput_mbps': {
            'mean': float(np.mean(tp_arr)),
            'std': float(np.std(tp_arr)),
            'ci95': float(1.96 * np.std(tp_arr, ddof=1) / np.sqrt(len(tp_arr))),
        },
        'fairness_mean': float(np.mean(fairness_vals)),
        'coverage_mean': float(np.mean(coverage_vals)),
        'per_scenario': [
            {'scenario_id': s.id, 'latency_ms': lat, 'throughput': tp,
             'fairness': f, 'coverage': c}
            for s, lat, tp, f, c in zip(scenarios, latencies, throughputs,
                                         fairness_vals, coverage_vals)
        ],
    }

    print(f"\n  Results for {config_name}:")
    print(f"    Latency: {result['latency_ms']['mean']:.1f} +/- "
          f"{result['latency_ms']['std']:.1f} ms "
          f"(p50={result['latency_ms']['p50']:.1f}, "
          f"p95={result['latency_ms']['p95']:.1f})")
    print(f"    Throughput: {result['throughput_mbps']['mean']:.1f} +/- "
          f"{result['throughput_mbps']['ci95']:.1f} Mbps")
    print(f"    Fairness: {result['fairness_mean']:.3f}, "
          f"Coverage: {result['coverage_mean']*100:.1f}%")
    print(f"    Parse failures: {parse_failures}/{len(scenarios)}")

    return result


def main():
    scenarios = generate_scenarios(NUM_SCENARIOS)
    results = {'configs': []}

    # ---- Config 1-3: 4-bit quantized model ----
    print("\n" + "=" * 60)
    print("Loading 4-bit quantized model with LoRA adapters...")
    print("=" * 60)
    model_4bit, tok_4bit = load_model_4bit()

    # Config 1: Baseline
    r = benchmark_config(model_4bit, tok_4bit, scenarios,
                         'baseline_4bit', max_new_tokens=150,
                         temperature=0.3, do_sample=True)
    results['configs'].append(r)

    # Config 2: Reduced tokens
    r = benchmark_config(model_4bit, tok_4bit, scenarios,
                         'reduced_tokens_4bit', max_new_tokens=50,
                         temperature=0.3, do_sample=True)
    results['configs'].append(r)

    # Config 3: Greedy
    r = benchmark_config(model_4bit, tok_4bit, scenarios,
                         'greedy_4bit', max_new_tokens=50,
                         temperature=0.0, do_sample=False)
    results['configs'].append(r)

    # Free 4-bit model
    del model_4bit, tok_4bit
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Config 4: Merged FP16 ----
    print("\n" + "=" * 60)
    print("Loading merged FP16 model (LoRA adapters merged)...")
    print("=" * 60)
    model_fp16, tok_fp16 = load_model_merged_fp16()

    r = benchmark_config(model_fp16, tok_fp16, scenarios,
                         'merged_fp16_greedy', max_new_tokens=50,
                         temperature=0.0, do_sample=False)
    results['configs'].append(r)

    del model_fp16, tok_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25s} {'Mean(ms)':>10s} {'Std':>8s} "
          f"{'P50':>8s} {'P95':>8s} {'Thpt':>8s}")
    print("-" * 70)
    for c in results['configs']:
        l = c['latency_ms']
        print(f"{c['config_name']:<25s} {l['mean']:>10.1f} {l['std']:>8.1f} "
              f"{l['p50']:>8.1f} {l['p95']:>8.1f} "
              f"{c['throughput_mbps']['mean']:>8.1f}")

    path = save_results(results, 'latency_benchmark')
    print(f"\nDone! Results at {path}")


if __name__ == '__main__':
    main()
