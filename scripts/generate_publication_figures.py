#!/usr/bin/env python3
"""
Generate Publication-Quality Figures from SGAC Training Results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use non-interactive backend
matplotlib.use('Agg')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def load_data(results_dir):
    """Load training history and final results."""
    results_dir = Path(results_dir)

    with open(results_dir / 'history.json', 'r') as f:
        history = json.load(f)

    with open(results_dir / 'final_results.json', 'r') as f:
        final = json.load(f)

    return history, final

def analyze_convergence(history):
    """Analyze if training has converged."""
    episodes = np.array(history['episode'])
    throughputs = np.array(history['episode_throughput'])

    # Smooth throughput with moving average
    window = 50
    smoothed = np.convolve(throughputs, np.ones(window)/window, mode='valid')

    # Check last 20% of training
    last_portion = smoothed[-len(smoothed)//5:]

    # Convergence metrics
    mean_last = np.mean(last_portion)
    std_last = np.std(last_portion)
    cv = std_last / mean_last  # Coefficient of variation

    # Check for trend in last portion
    x = np.arange(len(last_portion))
    slope, _ = np.polyfit(x, last_portion, 1)

    print("\n=== Convergence Analysis ===")
    print(f"Final throughput (last 20%): {mean_last:.2f} +/- {std_last:.2f} Mbps")
    print(f"Coefficient of variation: {cv:.4f}")
    print(f"Trend slope: {slope:.4f} Mbps/episode")

    if cv < 0.05 and abs(slope) < 0.1:
        print("Status: CONVERGED (stable with minimal trend)")
        converged = True
    elif cv < 0.1:
        print("Status: MOSTLY CONVERGED (low variance)")
        converged = True
    else:
        print("Status: MAY BENEFIT FROM MORE TRAINING")
        converged = False

    return converged, mean_last, std_last

def fig1_convergence(history, final, output_dir):
    """Figure 1: Training convergence comparison."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    episodes = np.array(history['episode'])
    throughputs = np.array(history['episode_throughput'])
    sca_throughputs = np.array(history['sca_throughput'])

    # Smooth curves
    window = 20
    smoothed_sgac = np.convolve(throughputs, np.ones(window)/window, mode='valid')
    smoothed_sca = np.convolve(sca_throughputs, np.ones(window)/window, mode='valid')
    eps_smooth = episodes[window-1:]

    # Plot
    ax.plot(eps_smooth, smoothed_sgac, 'b-', linewidth=2, label='SGAC (Ours)')
    ax.plot(eps_smooth, smoothed_sca, 'g--', linewidth=2, label='SCA Baseline')

    # Add baseline references
    ax.axhline(y=final['baselines']['analytical']['mean'], color='orange',
               linestyle=':', linewidth=1.5, label='Analytical')
    ax.axhline(y=final['baselines']['random']['mean'], color='gray',
               linestyle=':', linewidth=1.5, label='Random')

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_title('SGAC Training Convergence')
    ax.legend(loc='lower right')
    ax.set_xlim([0, max(episodes)])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_convergence.pdf')
    plt.savefig(output_dir / 'fig1_convergence.png')
    plt.close()
    print("Generated: fig1_convergence.pdf/png")

def fig2_throughput_comparison(final, output_dir):
    """Figure 2: Bar chart comparing all methods."""
    fig, ax = plt.subplots(figsize=(6, 4))

    methods = ['Random', 'Analytical', 'SCA-20', 'SGAC\n(Ours)']
    means = [
        final['baselines']['random']['mean'],
        final['baselines']['analytical']['mean'],
        final['baselines']['sca']['mean'],
        final['sgac']['mean_throughput']
    ]
    stds = [
        final['baselines']['random']['std'],
        final['baselines']['analytical']['std'],
        final['baselines']['sca']['std'],
        final['sgac']['std_throughput']
    ]

    colors = ['#808080', '#FFA500', '#2E8B57', '#DC143C']

    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{mean:.0f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Throughput (Mbps)')
    ax.set_title('Throughput Comparison Across Methods')
    ax.set_ylim([0, max(means) * 1.15])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_throughput_comparison.pdf')
    plt.savefig(output_dir / 'fig2_throughput_comparison.png')
    plt.close()
    print("Generated: fig2_throughput_comparison.pdf/png")

def fig3_reward_curve(history, output_dir):
    """Figure 3: Episode reward during training."""
    fig, ax = plt.subplots(figsize=(7, 4))

    episodes = np.array(history['episode'])
    rewards = np.array(history['episode_reward'])

    # Smooth
    window = 30
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    eps_smooth = episodes[window-1:]

    ax.plot(eps_smooth, smoothed, 'r-', linewidth=2)
    ax.fill_between(eps_smooth, smoothed - np.std(rewards[:len(smoothed)])*0.5,
                    smoothed + np.std(rewards[:len(smoothed)])*0.5, alpha=0.2, color='red')

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('SGAC Learning Progress')
    ax.set_xlim([0, max(episodes)])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_reward_curve.pdf')
    plt.savefig(output_dir / 'fig3_reward_curve.png')
    plt.close()
    print("Generated: fig3_reward_curve.pdf/png")

def fig4_floor_activations(history, output_dir):
    """Figure 4: Floor mechanism activation over training."""
    fig, ax = plt.subplots(figsize=(7, 4))

    episodes = np.array(history['episode'])
    floor_acts = np.array(history['floor_activations'])

    # Smooth
    window = 30
    smoothed = np.convolve(floor_acts, np.ones(window)/window, mode='valid')
    eps_smooth = episodes[window-1:]

    ax.plot(eps_smooth, smoothed, 'purple', linewidth=2)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1, label='Target (0 = all improvements)')

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Floor Activations per Episode')
    ax.set_title('Performance Floor Mechanism Activity')
    ax.set_xlim([0, max(episodes)])
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_floor_activations.pdf')
    plt.savefig(output_dir / 'fig4_floor_activations.png')
    plt.close()
    print("Generated: fig4_floor_activations.pdf/png")

def fig5_loss_curves(history, output_dir):
    """Figure 5: Critic and actor loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    episodes = np.array(history['episode'])
    critic_loss = np.array(history['critic_loss'])
    actor_loss = np.array(history['actor_loss'])

    # Filter out zeros (warmup period)
    mask = critic_loss > 0

    # Smooth
    window = 30
    if sum(mask) > window:
        critic_smooth = np.convolve(critic_loss[mask], np.ones(window)/window, mode='valid')
        actor_smooth = np.convolve(actor_loss[mask], np.ones(window)/window, mode='valid')
        eps_smooth = episodes[mask][window-1:]

        ax1.plot(eps_smooth, critic_smooth, 'b-', linewidth=2)
        ax1.set_xlabel('Training Episode')
        ax1.set_ylabel('Critic Loss')
        ax1.set_title('Twin Critic Loss')
        ax1.set_yscale('log')

        ax2.plot(eps_smooth, actor_smooth, 'r-', linewidth=2)
        ax2.set_xlabel('Training Episode')
        ax2.set_ylabel('Actor Loss')
        ax2.set_title('Actor Loss')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_loss_curves.pdf')
    plt.savefig(output_dir / 'fig5_loss_curves.png')
    plt.close()
    print("Generated: fig5_loss_curves.pdf/png")

def fig6_exploration_noise(history, output_dir):
    """Figure 6: Exploration noise decay."""
    fig, ax = plt.subplots(figsize=(6, 4))

    episodes = np.array(history['episode'])
    noise = np.array(history['exploration_noise'])

    ax.plot(episodes, noise, 'g-', linewidth=2)
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Exploration Noise σ')
    ax.set_title('Exploration Noise Annealing')
    ax.set_xlim([0, max(episodes)])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_exploration_noise.pdf')
    plt.savefig(output_dir / 'fig6_exploration_noise.png')
    plt.close()
    print("Generated: fig6_exploration_noise.pdf/png")

def fig7_improvement_over_sca(history, output_dir):
    """Figure 7: Improvement percentage over SCA during training."""
    fig, ax = plt.subplots(figsize=(7, 4))

    episodes = np.array(history['episode'])
    throughputs = np.array(history['episode_throughput'])
    sca_throughputs = np.array(history['sca_throughput'])

    # Calculate improvement percentage
    improvement_pct = (throughputs - sca_throughputs) / sca_throughputs * 100

    # Smooth
    window = 30
    smoothed = np.convolve(improvement_pct, np.ones(window)/window, mode='valid')
    eps_smooth = episodes[window-1:]

    ax.plot(eps_smooth, smoothed, 'b-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='SCA Baseline')
    ax.fill_between(eps_smooth, 0, smoothed, where=smoothed>0, alpha=0.3, color='green', label='Improvement')
    ax.fill_between(eps_smooth, 0, smoothed, where=smoothed<0, alpha=0.3, color='red', label='Below SCA')

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Improvement over SCA (%)')
    ax.set_title('SGAC Performance Relative to SCA Baseline')
    ax.set_xlim([0, max(episodes)])
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_improvement_over_sca.pdf')
    plt.savefig(output_dir / 'fig7_improvement_over_sca.png')
    plt.close()
    print("Generated: fig7_improvement_over_sca.pdf/png")

def generate_summary_table(final, output_dir):
    """Generate LaTeX table for paper."""
    table = r"""
\begin{table}[t]
\centering
\caption{Throughput Performance Comparison}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Mean (Mbps)} & \textbf{Std} & \textbf{vs SGAC} \\
\midrule
Random & %.1f & %.1f & -%.1f\%% \\
Analytical & %.1f & %.1f & -%.1f\%% \\
SCA-20 & %.1f & %.1f & -%.1f\%% \\
\midrule
\textbf{SGAC (Ours)} & \textbf{%.1f} & \textbf{%.1f} & -- \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        final['baselines']['random']['mean'],
        final['baselines']['random']['std'],
        (final['sgac']['mean_throughput'] - final['baselines']['random']['mean']) / final['sgac']['mean_throughput'] * 100,
        final['baselines']['analytical']['mean'],
        final['baselines']['analytical']['std'],
        (final['sgac']['mean_throughput'] - final['baselines']['analytical']['mean']) / final['sgac']['mean_throughput'] * 100,
        final['baselines']['sca']['mean'],
        final['baselines']['sca']['std'],
        (final['sgac']['mean_throughput'] - final['baselines']['sca']['mean']) / final['sgac']['mean_throughput'] * 100,
        final['sgac']['mean_throughput'],
        final['sgac']['std_throughput']
    )

    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write(table)
    print("Generated: results_table.tex")

def main():
    # Paths
    results_dir = Path('/home/it-services/ros2_ws/src/vla_6g_tvt/results_vastai/sgac_final')
    output_dir = Path('/home/it-services/ros2_ws/src/vla_6g_tvt/paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Publication Figures from SGAC Training Results")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    history, final = load_data(results_dir)
    print(f"Loaded {len(history['episode'])} episodes")

    # Analyze convergence
    converged, mean_final, std_final = analyze_convergence(history)

    # Generate figures
    print("\nGenerating figures...")
    fig1_convergence(history, final, output_dir)
    fig2_throughput_comparison(final, output_dir)
    fig3_reward_curve(history, output_dir)
    fig4_floor_activations(history, output_dir)
    fig5_loss_curves(history, output_dir)
    fig6_exploration_noise(history, output_dir)
    fig7_improvement_over_sca(history, output_dir)

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_summary_table(final, output_dir)

    print("\n" + "=" * 60)
    print("All figures saved to:", output_dir)
    print("=" * 60)

    # Print summary
    print("\n=== TRAINING SUMMARY ===")
    print(f"Episodes: {len(history['episode'])}")
    print(f"Training time: {final['training_time_minutes']:.1f} minutes")
    print(f"Final SGAC throughput: {final['sgac']['mean_throughput']:.2f} +/- {final['sgac']['std_throughput']:.2f} Mbps")
    print(f"Improvement vs SCA: +{final['sgac']['mean_improvement_pct']:.2f}%")
    print(f"Floor guarantee: {final['sgac']['floor_guarantee_rate']*100:.0f}%")

    if converged:
        print("\n✓ Training has CONVERGED - 2000 episodes is sufficient")
    else:
        print("\n⚠ Training may benefit from more episodes")

    return converged

if __name__ == '__main__':
    main()
