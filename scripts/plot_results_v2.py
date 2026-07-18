#!/usr/bin/env python3
"""
Updated Publication Figures (v2) for VLA-6G TVT Paper

New figures:
  1. latency_optimization.png — bar chart of latency across M1 configs
  2. throughput_7methods.png — all 7 methods (+ MLP, PPO) compared
  3. mobility_vs_speed.png — throughput degradation by speed per method
  4. pareto_updated.png — Pareto frontier with real latency + MLP + PPO

Also regenerates the original figures with all 7 methods.
"""

import json
import sys
import os
import glob
import shutil
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = '/home/it-services/ros2_ws/src/vla_6g_tvt/results'
FIGURES_DIR = '/home/it-services/ros2_ws/src/vla_6g_tvt/paper/figures'

# 7 methods
METHOD_COLORS = {
    'random': '#999999',
    'static': '#5DA5DA',
    'analytical': '#FAA43A',
    'mlp': '#B276B2',
    'ppo': '#8ECA6C',
    'vla': '#F15854',
    'optimized': '#60BD68',
}
METHOD_ORDER = ['random', 'static', 'analytical', 'mlp', 'ppo', 'vla', 'optimized']
METHOD_LABELS = {
    'random': 'Random',
    'static': 'Static',
    'analytical': 'Analytical',
    'mlp': 'MLP',
    'ppo': 'PPO',
    'vla': 'VLA (Ours)',
    'optimized': 'Optimized',
}


def load_latest(pattern):
    """Load latest JSON matching pattern in RESULTS_DIR."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Latency Optimization (M1)
# ---------------------------------------------------------------------------

def plot_latency_optimization(lat_data):
    """Bar chart of latency across M1 benchmark configs (broken y-axis)."""
    if lat_data is None:
        print("SKIP: No latency benchmark data found")
        return

    configs = lat_data['configs']
    names = [c['config_name'].replace('_', '\n') for c in configs]
    means = [c['latency_ms']['mean'] for c in configs]
    stds = [c['latency_ms']['std'] for c in configs]
    p95s = [c['latency_ms']['p95'] for c in configs]

    # Broken y-axis: top panel for baseline, bottom panel for optimized configs
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 5.5),
        gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.07})

    x = np.arange(len(names))
    colors = ['#5DA5DA', '#FAA43A', '#F15854', '#60BD68']

    # Draw bars and error bars on both axes
    for ax in [ax_top, ax_bot]:
        ax.bar(x, means, yerr=stds, capsize=5,
               color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.scatter(x, p95s, marker='v', color='red', zorder=5, s=50)
        ax.grid(axis='y', alpha=0.3)

    # Y-axis limits: break between 4500 and 8000
    ax_top.set_ylim(8000, 21000)
    ax_bot.set_ylim(0, 4500)

    # Hide spines at the break
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)

    # Draw diagonal break marks
    d = 0.015
    bkw = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=1)
    ax_top.plot((-d, +d), (-d*2, +d*2), **bkw)
    ax_top.plot((1-d, 1+d), (-d*2, +d*2), **bkw)
    bkw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1-d*2, 1+d*2), **bkw)
    ax_bot.plot((1-d, 1+d), (1-d*2, 1+d*2), **bkw)

    # Value label + P95 for baseline (top panel)
    ax_top.text(0, p95s[0] + 500, f'{means[0]:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_top.legend(['P95'], loc='upper right',
                  handler_map={str: None},
                  handles=[plt.Line2D([], [], marker='v', color='red',
                                      linestyle='None', markersize=7)])

    # Value labels for bars 1-3 (bottom panel) — placed well above P95 markers
    for i in range(1, len(means)):
        top = max(means[i] + stds[i], p95s[i])
        ax_bot.text(i, top + 350, f'{means[i]:.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # X-axis labels
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(names, fontsize=9)

    # Shared y-label
    fig.text(0.02, 0.5, 'Inference Latency (ms)',
             va='center', rotation='vertical', fontsize=11)
    ax_top.set_title('VLA Inference Latency by Configuration')

    # Speedup annotation — small arrow to the right of the last bar
    if len(means) >= 2:
        speedup = means[0] / means[-1]
        last_top = max(means[-1] + stds[-1], p95s[-1])
        # Text to the right with a short downward arrow pointing at bar top
        ax_bot.annotate(f'{speedup:.1f}$\\times$\nspeedup',
                        xy=(len(means)-1 + 0.35, last_top + 100),
                        xytext=(len(means)-1 + 0.35, 2400),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        fontsize=10, fontweight='bold', ha='center',
                        va='bottom', color='red')

    fig.savefig(os.path.join(FIGURES_DIR, 'latency_optimization.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved latency_optimization.png")
    return configs


# ---------------------------------------------------------------------------
# Figure 2: Throughput 7 Methods
# ---------------------------------------------------------------------------

def plot_throughput_7methods(eval_data, mlp_data, ppo_data):
    """Bar chart: throughput for all 7 methods with 95% CI."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    means, cis, colors, labels = [], [], [], []

    # Original 5 from evaluation data
    if eval_data is not None:
        for m in ['random', 'static', 'analytical', 'vla', 'optimized']:
            results = [r for r in eval_data['results'] if r['method'] == m]
            if not results:
                continue
            tp = [r['total_throughput'] for r in results]
            arr = np.array(tp)
            mean = np.mean(arr)
            se = np.std(arr, ddof=1) / np.sqrt(len(arr))

            # Insert MLP/PPO after analytical
            if m == 'analytical':
                means.append(mean)
                cis.append(1.96 * se)
                colors.append(METHOD_COLORS[m])
                labels.append(METHOD_LABELS[m])

                # MLP
                if mlp_data is not None:
                    s = mlp_data['summary']['throughput_mbps']
                    means.append(s['mean'])
                    cis.append(s['ci95'])
                    colors.append(METHOD_COLORS['mlp'])
                    labels.append(METHOD_LABELS['mlp'])

                # PPO
                if ppo_data is not None:
                    s = ppo_data['summary']['throughput_mbps']
                    means.append(s['mean'])
                    cis.append(s['ci95'])
                    colors.append(METHOD_COLORS['ppo'])
                    labels.append(METHOD_LABELS['ppo'])
            else:
                means.append(mean)
                cis.append(1.96 * se)
                colors.append(METHOD_COLORS[m])
                labels.append(METHOD_LABELS[m])

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, color=colors, capsize=5,
           edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (m, c) in enumerate(zip(means, cis)):
        ax.text(i, m + c + 2, f'{m:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Total Throughput (Mbps)')
    ax.set_title('Throughput Comparison Across All Methods')
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(os.path.join(FIGURES_DIR, 'throughput_7methods.png'))
    plt.close(fig)
    print("Saved throughput_7methods.png")


# ---------------------------------------------------------------------------
# Figure 3: Mobility vs Speed
# ---------------------------------------------------------------------------

def plot_mobility_vs_speed(mob_data):
    """Line plot: throughput degradation by speed per method."""
    if mob_data is None:
        print("SKIP: No mobility data found")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    speeds = mob_data['speeds_kmh']
    methods = mob_data['methods']

    for method in methods:
        tp_means = []
        tp_cis = []
        for speed in speeds:
            key = f'{speed}_kmh'
            if key in mob_data['results'] and method in mob_data['results'][key]:
                agg = mob_data['results'][key][method]['aggregate']
                tp_means.append(agg['mean_throughput'])
                tp_cis.append(agg.get('ci95_throughput', 0))
            else:
                tp_means.append(np.nan)
                tp_cis.append(0)

        color = METHOD_COLORS.get(method, '#333')
        label = METHOD_LABELS.get(method, method)
        ax.errorbar(speeds, tp_means, yerr=tp_cis, marker='o',
                     color=color, label=label, linewidth=2, markersize=6,
                     capsize=4)

    ax.set_xlabel('Vehicle Speed (km/h)')
    ax.set_ylabel('Mean Throughput (Mbps)')
    ax.set_title('Throughput Under Vehicular Mobility')
    ax.set_xticks(speeds)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(os.path.join(FIGURES_DIR, 'mobility_vs_speed.png'))
    plt.close(fig)
    print("Saved mobility_vs_speed.png")


# ---------------------------------------------------------------------------
# Figure 4: Updated Pareto Frontier
# ---------------------------------------------------------------------------

def plot_pareto_updated(eval_data, mlp_data, ppo_data, lat_data):
    """Scatter: quality (throughput) vs latency for all 7 methods."""
    fig, ax = plt.subplots(figsize=(7, 5))

    points = []

    # Original methods from eval_data
    if eval_data is not None:
        for m in ['random', 'static', 'analytical', 'optimized']:
            results = [r for r in eval_data['results'] if r['method'] == m]
            if not results:
                continue
            tp = np.mean([r['total_throughput'] for r in results])
            lat = np.mean([r.get('inference_time_ms', 0.01) for r in results])
            lat = max(lat, 0.01)  # Avoid log(0)
            points.append((m, tp, lat))

    # VLA — use real latency from M1 (merged_fp16_greedy) if available
    if eval_data is not None:
        vla_results = [r for r in eval_data['results'] if r['method'] == 'vla']
        if vla_results:
            vla_tp = np.mean([r['total_throughput'] for r in vla_results])
            vla_lat = np.mean([r.get('inference_time_ms', 9800) for r in vla_results])
            # Override with M1 measured latency
            if lat_data is not None:
                for cfg in lat_data.get('configs', []):
                    if cfg['config_name'] == 'merged_fp16_greedy':
                        vla_lat = cfg['latency_ms']['mean']
                        break
            points.append(('vla', vla_tp, vla_lat))

    # MLP
    if mlp_data is not None:
        tp = mlp_data['summary']['throughput_mbps']['mean']
        lat = mlp_data['summary']['latency_ms']['mean']
        points.append(('mlp', tp, lat))

    # PPO
    if ppo_data is not None:
        tp = ppo_data['summary']['throughput_mbps']['mean']
        lat = ppo_data['summary']['latency_ms']['mean']
        points.append(('ppo', tp, lat))

    for m, tp, lat in points:
        color = METHOD_COLORS.get(m, '#333')
        label = METHOD_LABELS.get(m, m)
        marker = '*' if m == 'vla' else 'o'
        size = 200 if m == 'vla' else 100
        ax.scatter(lat, tp, c=color, s=size, marker=marker,
                   label=label, edgecolors='black', linewidth=0.5, zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Decision Latency (ms)')
    ax.set_ylabel('Mean Throughput (Mbps)')
    ax.set_title('Quality--Latency Pareto Frontier')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # Draw Pareto frontier line
    sorted_pts = sorted(points, key=lambda p: p[2])  # sort by latency
    pareto_pts = []
    max_tp = -1
    for m, tp, lat in sorted_pts:
        if tp > max_tp:
            pareto_pts.append((lat, tp))
            max_tp = tp
    if len(pareto_pts) >= 2:
        px, py = zip(*pareto_pts)
        ax.plot(px, py, 'k--', alpha=0.3, linewidth=1)

    fig.savefig(os.path.join(FIGURES_DIR, 'pareto_updated.png'))
    plt.close(fig)
    print("Saved pareto_updated.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load all available results
    eval_data = load_latest('evaluation_*.json')
    mlp_data = load_latest('mlp_baseline_*.json')
    ppo_data = load_latest('drl_baseline_*.json')
    lat_data = load_latest('latency_benchmark_*.json')
    mob_data = load_latest('mobility_evaluation_*.json')

    print("Data loaded:")
    print(f"  Evaluation:  {'YES' if eval_data else 'NO'}")
    print(f"  MLP:         {'YES' if mlp_data else 'NO'}")
    print(f"  PPO:         {'YES' if ppo_data else 'NO'}")
    print(f"  Latency:     {'YES' if lat_data else 'NO'}")
    print(f"  Mobility:    {'YES' if mob_data else 'NO'}")

    # Generate all figures
    plot_latency_optimization(lat_data)
    plot_throughput_7methods(eval_data, mlp_data, ppo_data)
    plot_mobility_vs_speed(mob_data)
    plot_pareto_updated(eval_data, mlp_data, ppo_data, lat_data)

    # Copy to results dir too
    for fname in os.listdir(FIGURES_DIR):
        if fname.endswith('.png'):
            src = os.path.join(FIGURES_DIR, fname)
            dst = os.path.join(RESULTS_DIR, fname)
            shutil.copy2(src, dst)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
