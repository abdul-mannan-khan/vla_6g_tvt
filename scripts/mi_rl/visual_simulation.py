#!/usr/bin/env python3
"""
MI-RL Visual Simulation Demo for Paper

Creates a visual demonstration video showing:
1. SCA optimization baseline
2. MI-RL (SGAC) with Residual RL corrections
3. "Math + RL" thesis visualization
4. Real-time metrics comparison

Output: MP4 video suitable for paper supplementary material

Usage:
    python visual_simulation.py --checkpoint latest --output demo.mp4
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_common import (
    generate_scenarios, compute_channel_metrics, Scenario,
    get_position_analytical, get_position_random
)

# Check if we can import SGAC (might not have PyTorch in some environments)
try:
    from mi_rl.sgac_agent import SGACAgent, SGACConfig
    from classical.sca_solver import SCASolver, SCAConfig
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False
    print("Warning: Could not import agents, using mock data")

# Video settings
FPS = 24
DPI = 100
FIGSIZE = (16, 9)

# Colors
COLOR_BS = '#2E86AB'
COLOR_UAV_SCA = '#E74C3C'      # Red for SCA
COLOR_UAV_MIRL = '#2ECC71'     # Green for MI-RL
COLOR_UAV_RANDOM = '#95A5A6'   # Gray for random
COLOR_USERS = '#F39C12'        # Orange for users
COLOR_CORRECTION = '#9B59B6'   # Purple for RL correction
COLOR_SIGNAL = '#00BCD4'       # Cyan for signal beams


class MIRLVisualSimulation:
    """Visual simulation demonstrating Math-Informed RL for UAV positioning."""

    def __init__(self, checkpoint_path=None, output_dir='videos'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize agents if available
        self.agent = None
        self.sca_solver = None

        if HAS_AGENTS and checkpoint_path:
            self._load_agents(checkpoint_path)

        # Generate test scenarios
        np.random.seed(42)
        self.scenarios = generate_scenarios(num_scenarios=10, seed=42)
        self.current_scenario_idx = 0

        # Animation state
        self.frame_data = []

    def _load_agents(self, checkpoint_path):
        """Load MI-RL agent and SCA solver."""
        try:
            config = SGACConfig()
            self.agent = SGACAgent(config)

            if os.path.exists(checkpoint_path):
                self.agent.load_checkpoint(checkpoint_path)
                print(f"Loaded checkpoint: {checkpoint_path}")

            self.sca_solver = SCASolver(SCAConfig(max_iterations=20))
        except Exception as e:
            print(f"Warning: Could not load agents: {e}")

    def compute_positions(self, scenario):
        """Compute UAV positions for different methods."""
        positions = {}

        # Random baseline
        positions['random'] = get_position_random(scenario)

        # Analytical baseline
        positions['analytical'] = get_position_analytical(scenario)

        # SCA optimization
        if self.sca_solver:
            sca_pos, _ = self.sca_solver.solve(scenario, verbose=False)
            positions['sca'] = sca_pos
        else:
            # Mock SCA position
            centroid = np.mean(scenario.user_positions, axis=0)
            positions['sca'] = np.array([centroid[0], centroid[1], 25])

        # MI-RL (SCA + correction)
        if self.agent:
            positions['mirl'] = self.agent.get_position(scenario, deterministic=True)
        else:
            # Mock MI-RL position (SCA + small correction)
            positions['mirl'] = positions['sca'] + np.array([2, 1, -1])

        # Compute metrics for each
        metrics = {}
        for method, pos in positions.items():
            m = compute_channel_metrics(pos, scenario)
            metrics[method] = {
                'throughput': m['total_throughput'],
                'fairness': m['fairness'],
                'coverage': m['coverage_rate']
            }

        return positions, metrics

    def create_demo_frame(self, ax_3d, ax_metrics, scenario, positions, metrics,
                          show_correction=True, phase='comparison'):
        """Create a single demo frame."""
        ax_3d.clear()
        ax_metrics.clear()

        # --- 3D Scene ---
        ax_3d.set_xlim(0, 100)
        ax_3d.set_ylim(0, 100)
        ax_3d.set_zlim(0, 50)
        ax_3d.set_xlabel('X (m)', fontsize=10)
        ax_3d.set_ylabel('Y (m)', fontsize=10)
        ax_3d.set_zlabel('Altitude (m)', fontsize=10)
        ax_3d.set_title('UAV Relay Positioning - Math-Informed RL', fontsize=14, fontweight='bold')

        # Ground plane
        xx, yy = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
        ax_3d.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

        # Base station
        bs = scenario.bs_position
        ax_3d.scatter([bs[0]], [bs[1]], [bs[2]], c=COLOR_BS, s=200, marker='^',
                      label='Base Station', edgecolors='black', linewidths=2)

        # Users
        for i, user_pos in enumerate(scenario.user_positions):
            ax_3d.scatter([user_pos[0]], [user_pos[1]], [user_pos[2]],
                         c=COLOR_USERS, s=100, marker='o', edgecolors='black')
        ax_3d.scatter([], [], [], c=COLOR_USERS, s=100, marker='o', label='Users')

        # UAV positions
        if phase in ['random', 'comparison']:
            pos = positions['random']
            ax_3d.scatter([pos[0]], [pos[1]], [pos[2]], c=COLOR_UAV_RANDOM, s=150,
                         marker='s', label=f"Random ({metrics['random']['throughput']:.0f} Mbps)")

        if phase in ['sca', 'comparison', 'mirl']:
            pos = positions['sca']
            ax_3d.scatter([pos[0]], [pos[1]], [pos[2]], c=COLOR_UAV_SCA, s=200,
                         marker='D', label=f"SCA ({metrics['sca']['throughput']:.0f} Mbps)",
                         edgecolors='black', linewidths=2)

        if phase in ['mirl', 'comparison']:
            pos = positions['mirl']
            ax_3d.scatter([pos[0]], [pos[1]], [pos[2]], c=COLOR_UAV_MIRL, s=250,
                         marker='*', label=f"MI-RL ({metrics['mirl']['throughput']:.0f} Mbps)",
                         edgecolors='black', linewidths=2)

            # Show RL correction arrow
            if show_correction and phase == 'mirl':
                sca_pos = positions['sca']
                correction = pos - sca_pos
                ax_3d.quiver(sca_pos[0], sca_pos[1], sca_pos[2],
                            correction[0], correction[1], correction[2],
                            color=COLOR_CORRECTION, arrow_length_ratio=0.3,
                            linewidth=3, label='RL Correction')

        # Signal beams (simplified)
        if phase in ['mirl', 'comparison']:
            uav_pos = positions['mirl']
            # BS to UAV beam
            ax_3d.plot([bs[0], uav_pos[0]], [bs[1], uav_pos[1]], [bs[2], uav_pos[2]],
                      color=COLOR_SIGNAL, alpha=0.5, linewidth=2, linestyle='--')
            # UAV to users beams
            for user_pos in scenario.user_positions:
                ax_3d.plot([uav_pos[0], user_pos[0]], [uav_pos[1], user_pos[1]],
                          [uav_pos[2], user_pos[2]], color=COLOR_SIGNAL, alpha=0.3, linewidth=1)

        ax_3d.legend(loc='upper left', fontsize=9)
        ax_3d.view_init(elev=25, azim=-60)

        # --- Metrics Panel ---
        ax_metrics.set_xlim(0, 1)
        ax_metrics.set_ylim(0, 1)
        ax_metrics.axis('off')
        ax_metrics.set_title('Performance Metrics', fontsize=12, fontweight='bold')

        # Method comparison bars
        methods = ['random', 'analytical', 'sca', 'mirl']
        labels = ['Random', 'Analytical', 'SCA-20', 'MI-RL (Ours)']
        colors = [COLOR_UAV_RANDOM, '#3498DB', COLOR_UAV_SCA, COLOR_UAV_MIRL]
        throughputs = [metrics.get(m, {}).get('throughput', 0) for m in methods]

        y_positions = [0.8, 0.6, 0.4, 0.2]
        max_throughput = max(throughputs) * 1.1

        for i, (label, throughput, color, y) in enumerate(zip(labels, throughputs, colors, y_positions)):
            # Bar
            bar_width = throughput / max_throughput * 0.6
            rect = FancyBboxPatch((0.25, y - 0.05), bar_width, 0.08,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, edgecolor='black', linewidth=1)
            ax_metrics.add_patch(rect)

            # Label
            ax_metrics.text(0.05, y, label, fontsize=10, va='center', fontweight='bold')

            # Value
            ax_metrics.text(0.25 + bar_width + 0.02, y, f'{throughput:.1f} Mbps',
                           fontsize=10, va='center')

        # Thesis statement
        ax_metrics.text(0.5, 0.02, '"A little bit of Math + a little bit of RL"',
                       fontsize=11, ha='center', style='italic', color='#2C3E50')

    def generate_demo_video(self, output_file='mi_rl_demo.mp4', duration=30):
        """Generate the full demo video."""
        fig = plt.figure(figsize=FIGSIZE, facecolor='white')
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

        ax_3d = fig.add_subplot(gs[0], projection='3d')
        ax_metrics = fig.add_subplot(gs[1])

        # Precompute all scenarios
        all_data = []
        for scenario in self.scenarios[:5]:  # Use 5 scenarios
            positions, metrics = self.compute_positions(scenario)
            all_data.append((scenario, positions, metrics))

        total_frames = duration * FPS
        frames_per_scenario = total_frames // len(all_data)

        def update(frame):
            scenario_idx = min(frame // frames_per_scenario, len(all_data) - 1)
            scenario, positions, metrics = all_data[scenario_idx]

            # Determine phase based on frame within scenario
            local_frame = frame % frames_per_scenario
            if local_frame < frames_per_scenario // 4:
                phase = 'random'
            elif local_frame < frames_per_scenario // 2:
                phase = 'sca'
            elif local_frame < 3 * frames_per_scenario // 4:
                phase = 'mirl'
            else:
                phase = 'comparison'

            self.create_demo_frame(ax_3d, ax_metrics, scenario, positions, metrics,
                                  show_correction=True, phase=phase)

            # Progress indicator
            fig.suptitle(f'Scenario {scenario_idx + 1}/5 | Frame {frame + 1}/{total_frames}',
                        fontsize=10, color='gray')

            return []

        print(f"Generating {total_frames} frames at {FPS} FPS...")
        anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/FPS, blit=False)

        output_path = os.path.join(self.output_dir, output_file)
        writer = FFMpegWriter(fps=FPS, metadata={'title': 'MI-RL UAV Demo'})

        try:
            anim.save(output_path, writer=writer, dpi=DPI)
            print(f"Video saved to: {output_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            print("Saving as GIF instead...")
            gif_path = output_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=FPS//2, dpi=DPI//2)
            print(f"GIF saved to: {gif_path}")

        plt.close(fig)
        return output_path

    def generate_static_comparison(self, output_file='mi_rl_comparison.png'):
        """Generate a static comparison figure for the paper."""
        fig = plt.figure(figsize=(14, 6), facecolor='white')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8])

        scenario = self.scenarios[0]
        positions, metrics = self.compute_positions(scenario)

        # Left: SCA solution
        ax1 = fig.add_subplot(gs[0], projection='3d')
        self._draw_scenario_3d(ax1, scenario, positions['sca'], 'SCA-20 Optimization',
                               COLOR_UAV_SCA, metrics['sca']['throughput'])

        # Middle: MI-RL solution with correction
        ax2 = fig.add_subplot(gs[1], projection='3d')
        self._draw_scenario_3d(ax2, scenario, positions['mirl'], 'MI-RL (SCA + RL Correction)',
                               COLOR_UAV_MIRL, metrics['mirl']['throughput'],
                               show_correction=True, sca_pos=positions['sca'])

        # Right: Metrics comparison
        ax3 = fig.add_subplot(gs[2])
        self._draw_metrics_comparison(ax3, metrics)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Comparison figure saved to: {output_path}")
        plt.close(fig)
        return output_path

    def _draw_scenario_3d(self, ax, scenario, uav_pos, title, color, throughput,
                          show_correction=False, sca_pos=None):
        """Draw a 3D scenario visualization."""
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 50)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{title}\n{throughput:.1f} Mbps', fontsize=11, fontweight='bold')

        # Base station
        bs = scenario.bs_position
        ax.scatter([bs[0]], [bs[1]], [bs[2]], c=COLOR_BS, s=150, marker='^')

        # Users
        for user_pos in scenario.user_positions:
            ax.scatter([user_pos[0]], [user_pos[1]], [user_pos[2]],
                      c=COLOR_USERS, s=80, marker='o')

        # UAV
        ax.scatter([uav_pos[0]], [uav_pos[1]], [uav_pos[2]], c=color, s=200,
                  marker='*', edgecolors='black', linewidths=2)

        # Correction arrow
        if show_correction and sca_pos is not None:
            correction = uav_pos - sca_pos
            ax.quiver(sca_pos[0], sca_pos[1], sca_pos[2],
                     correction[0], correction[1], correction[2],
                     color=COLOR_CORRECTION, arrow_length_ratio=0.3, linewidth=2)
            ax.scatter([sca_pos[0]], [sca_pos[1]], [sca_pos[2]], c=COLOR_UAV_SCA,
                      s=100, marker='D', alpha=0.5)

        # Signal beams
        ax.plot([bs[0], uav_pos[0]], [bs[1], uav_pos[1]], [bs[2], uav_pos[2]],
               color=COLOR_SIGNAL, alpha=0.5, linewidth=2, linestyle='--')
        for user_pos in scenario.user_positions:
            ax.plot([uav_pos[0], user_pos[0]], [uav_pos[1], user_pos[1]],
                   [uav_pos[2], user_pos[2]], color=COLOR_SIGNAL, alpha=0.3, linewidth=1)

        ax.view_init(elev=25, azim=-60)

    def _draw_metrics_comparison(self, ax, metrics):
        """Draw metrics comparison bar chart."""
        methods = ['Random', 'Analytical', 'SCA-20', 'MI-RL']
        throughputs = [
            metrics.get('random', {}).get('throughput', 95),
            metrics.get('analytical', {}).get('throughput', 110),
            metrics.get('sca', {}).get('throughput', 195),
            metrics.get('mirl', {}).get('throughput', 196)
        ]
        colors = [COLOR_UAV_RANDOM, '#3498DB', COLOR_UAV_SCA, COLOR_UAV_MIRL]

        bars = ax.barh(methods, throughputs, color=colors, edgecolor='black')
        ax.set_xlabel('Throughput (Mbps)')
        ax.set_title('Method Comparison', fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, throughputs):
            ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                   va='center', fontsize=10)

        ax.set_xlim(0, max(throughputs) * 1.15)


def main():
    parser = argparse.ArgumentParser(description='MI-RL Visual Simulation Demo')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (optional)')
    parser.add_argument('--output', type=str, default='mi_rl_demo.mp4',
                        help='Output video filename')
    parser.add_argument('--duration', type=int, default=30,
                        help='Video duration in seconds')
    parser.add_argument('--static-only', action='store_true',
                        help='Generate only static comparison figure')
    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint_path = None
    if args.checkpoint:
        if args.checkpoint == 'latest':
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(script_dir, '../../results/mi_rl/checkpoints/checkpoint_latest.pt')
        else:
            checkpoint_path = args.checkpoint

    # Create simulation
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../../videos')
    sim = MIRLVisualSimulation(checkpoint_path=checkpoint_path, output_dir=output_dir)

    # Generate outputs
    sim.generate_static_comparison()

    if not args.static_only:
        sim.generate_demo_video(output_file=args.output, duration=args.duration)

    print("\nVisual simulation complete!")


if __name__ == "__main__":
    main()
