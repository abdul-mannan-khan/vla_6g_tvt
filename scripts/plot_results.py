#!/usr/bin/env python3
"""
Publication Figures for VLA-6G Evaluation Results

Generates:
1. Bar chart: throughput by method with error bars
2. Bar chart: latency comparison (log scale)
3. Scatter: VLA vs Optimized per-scenario
4. Grouped bars: performance by scenario type
"""

import json
import sys
import os
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

METHOD_COLORS = {
    'random': '#999999',
    'static': '#5DA5DA',
    'analytical': '#FAA43A',
    'vla': '#F15854',
    'optimized': '#60BD68',
}
METHOD_ORDER = ['random', 'static', 'analytical', 'vla', 'optimized']
METHOD_LABELS = {
    'random': 'Random',
    'static': 'Static',
    'analytical': 'Analytical',
    'vla': 'VLA (Ours)',
    'optimized': 'Optimized',
}


def load_latest_results(results_dir):
    """Load the most recent evaluation JSON."""
    files = sorted(glob.glob(os.path.join(results_dir, 'evaluation_*.json')))
    if not files:
        print(f"No evaluation files found in {results_dir}")
        sys.exit(1)
    path = files[-1]
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f)


def get_method_data(results, method):
    """Extract per-scenario results for a method."""
    return [r for r in results if r['method'] == method]


def plot_throughput_bars(data, methods, output_dir):
    """Bar chart: throughput by method with 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means, cis, colors, labels = [], [], [], []
    for m in methods:
        vals = [r['total_throughput'] for r in get_method_data(data['results'], m)]
        if not vals:
            continue
        arr = np.array(vals)
        mean = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(len(arr))
        means.append(mean)
        cis.append(1.96 * se)
        colors.append(METHOD_COLORS.get(m, '#333'))
        labels.append(METHOD_LABELS.get(m, m))

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, color=colors, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Total Throughput (Mbps)')
    ax.set_title('Throughput Comparison')
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'throughput_comparison.png'))
    plt.close(fig)
    print("Saved throughput_comparison.png")


def plot_latency_bars(data, methods, output_dir):
    """Bar chart: inference latency (log scale)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means, colors, labels = [], [], []
    for m in methods:
        vals = [r.get('inference_time_ms', 0) for r in get_method_data(data['results'], m)]
        if not vals:
            continue
        means.append(max(np.mean(vals), 0.01))  # avoid log(0)
        colors.append(METHOD_COLORS.get(m, '#333'))
        labels.append(METHOD_LABELS.get(m, m))

    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Latency (ms)')
    ax.set_yscale('log')
    ax.set_title('Inference Latency Comparison')
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'latency_comparison.png'))
    plt.close(fig)
    print("Saved latency_comparison.png")


def plot_vla_vs_optimized_scatter(data, output_dir):
    """Scatter: VLA throughput vs Optimized throughput per scenario."""
    vla = sorted(get_method_data(data['results'], 'vla'), key=lambda r: r['scenario_id'])
    opt = sorted(get_method_data(data['results'], 'optimized'), key=lambda r: r['scenario_id'])
    if not vla or not opt or len(vla) != len(opt):
        print("Skipping scatter: mismatched VLA/optimized results")
        return

    vla_tp = [r['total_throughput'] for r in vla]
    opt_tp = [r['total_throughput'] for r in opt]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(opt_tp, vla_tp, alpha=0.5, s=20, c=METHOD_COLORS['vla'], edgecolors='none')

    lims = [min(min(opt_tp), min(vla_tp)) * 0.9, max(max(opt_tp), max(vla_tp)) * 1.05]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='y=x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Optimized Throughput (Mbps)')
    ax.set_ylabel('VLA Throughput (Mbps)')
    ax.set_title('VLA vs Optimized (Per Scenario)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'vla_vs_optimized_scatter.png'))
    plt.close(fig)
    print("Saved vla_vs_optimized_scatter.png")


def plot_by_scenario_type(data, methods, output_dir):
    """Grouped bars: throughput by scenario type."""
    scenario_types = {0: 'Clustered', 1: 'Spread', 2: 'Line', 3: 'Circle'}
    present_methods = [m for m in methods if get_method_data(data['results'], m)]

    fig, ax = plt.subplots(figsize=(9, 5))
    n_types = len(scenario_types)
    n_methods = len(present_methods)
    bar_width = 0.15
    x = np.arange(n_types)

    for j, m in enumerate(present_methods):
        m_results = get_method_data(data['results'], m)
        means = []
        for type_id in scenario_types:
            vals = [r['total_throughput'] for r in m_results if r['scenario_id'] % 4 == type_id]
            means.append(np.mean(vals) if vals else 0)
        offset = (j - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, label=METHOD_LABELS.get(m, m),
               color=METHOD_COLORS.get(m, '#333'), edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(list(scenario_types.values()))
    ax.set_ylabel('Total Throughput (Mbps)')
    ax.set_title('Performance by Scenario Type')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'performance_by_scenario_type.png'))
    plt.close(fig)
    print("Saved performance_by_scenario_type.png")


def plot_throughput_cdf(data, methods, output_dir):
    """CDF of throughput for each method."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        vals = sorted([r['total_throughput'] for r in get_method_data(data['results'], m)])
        if not vals:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=METHOD_LABELS.get(m, m),
                color=METHOD_COLORS.get(m, '#333'), linewidth=2)
    ax.set_xlabel('Total Throughput (Mbps)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution of Throughput')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'throughput_cdf.png'))
    plt.close(fig)
    print("Saved throughput_cdf.png")


def plot_fairness_coverage_radar(data, methods, output_dir):
    """Multi-metric bar chart comparing fairness, coverage, and throughput normalized."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ('total_throughput', 'Throughput (Mbps)', axes[0]),
        ('fairness_index', 'Fairness Index', axes[1]),
        ('coverage_rate', 'Coverage Rate', axes[2]),
    ]
    for key, ylabel, ax in metrics:
        means, cis, colors, labels = [], [], [], []
        for m in methods:
            vals = [r[key] for r in get_method_data(data['results'], m)]
            if not vals:
                continue
            arr = np.array(vals)
            mean = np.mean(arr)
            se = np.std(arr, ddof=1) / np.sqrt(len(arr))
            means.append(mean)
            cis.append(1.96 * se)
            colors.append(METHOD_COLORS.get(m, '#333'))
            labels.append(METHOD_LABELS.get(m, m))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=cis, color=colors, capsize=4, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Multi-Metric Performance Comparison', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'multi_metric_comparison.png'))
    plt.close(fig)
    print("Saved multi_metric_comparison.png")


def main():
    results_dir = '/home/it-services/ros2_ws/src/vla_6g_tvt/results'
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    output_dir = results_dir
    figures_dir = '/home/it-services/ros2_ws/src/vla_6g_tvt/paper/figures'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    data = load_latest_results(results_dir)
    methods = [m for m in METHOD_ORDER if any(r['method'] == m for r in data['results'])]

    plot_throughput_bars(data, methods, output_dir)
    plot_latency_bars(data, methods, output_dir)
    plot_vla_vs_optimized_scatter(data, output_dir)
    plot_by_scenario_type(data, methods, output_dir)
    plot_throughput_cdf(data, methods, output_dir)
    plot_fairness_coverage_radar(data, methods, output_dir)

    # Copy to paper/figures for LaTeX
    import shutil
    for fname in ['throughput_comparison.png', 'latency_comparison.png',
                  'vla_vs_optimized_scatter.png', 'performance_by_scenario_type.png',
                  'throughput_cdf.png', 'multi_metric_comparison.png']:
        src = os.path.join(output_dir, fname)
        dst = os.path.join(figures_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print(f"\nAll figures saved to {output_dir} and {figures_dir}")


if __name__ == '__main__':
    main()
