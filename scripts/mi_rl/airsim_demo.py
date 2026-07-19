#!/usr/bin/env python3
"""
MI-RL AirSim Demo for UAV Relay Positioning

This script creates a visual demonstration video using AirSim simulation.
Shows the UAV moving to optimal relay positions computed by MI-RL.

Requirements:
    - AirSim running with a multirotor
    - Trained MI-RL checkpoint
    - OpenCV for video recording

Usage:
    python airsim_demo.py --checkpoint latest --duration 60 --output mi_rl_airsim_demo.mp4
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import AirSim
try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    print("Warning: AirSim not available. Install with: pip install airsim")

# Import MI-RL components
from eval_common import generate_scenarios, compute_channel_metrics, Scenario
from mi_rl.sgac_agent import SGACAgent, SGACConfig
from classical.sca_solver import SCASolver, SCAConfig


class MIRLAirSimDemo:
    """AirSim demonstration for MI-RL UAV relay positioning."""

    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path

        # Initialize agents
        self.agent = None
        self.sca_solver = None
        self._load_agents()

        # AirSim client
        self.client = None
        if AIRSIM_AVAILABLE:
            self._connect_airsim()

        # Video settings
        self.fps = 30
        self.resolution = (1920, 1080)
        self.video_writer = None

        # Demo state
        self.current_scenario_idx = 0
        self.scenarios = generate_scenarios(num_scenarios=5, seed=42)

        # Scaling: simulation uses 100x100m, AirSim uses different scale
        self.sim_to_airsim_scale = 1.0  # Adjust based on your AirSim environment

    def _load_agents(self):
        """Load MI-RL agent and SCA solver."""
        try:
            config = SGACConfig()
            self.agent = SGACAgent(config)

            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                self.agent.load_checkpoint(self.checkpoint_path)
                print(f"✅ Loaded MI-RL checkpoint: {self.checkpoint_path}")

            self.sca_solver = SCASolver(SCAConfig(max_iterations=20))
            print("✅ SCA solver initialized")

        except Exception as e:
            print(f"⚠️ Agent loading failed: {e}")

    def _connect_airsim(self):
        """Connect to AirSim simulator."""
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("✅ Connected to AirSim")
        except Exception as e:
            print(f"⚠️ AirSim connection failed: {e}")
            self.client = None

    def sim_to_airsim_pos(self, sim_pos):
        """Convert simulation position to AirSim coordinates."""
        # Simulation: (0-100, 0-100, 10-40) meters
        # AirSim: NED coordinates (North, East, Down)
        x = (sim_pos[0] - 50) * self.sim_to_airsim_scale  # Center at origin
        y = (sim_pos[1] - 50) * self.sim_to_airsim_scale
        z = -sim_pos[2] * self.sim_to_airsim_scale  # NED: negative is up
        return airsim.Vector3r(x, y, z)

    def move_to_position(self, target_pos, velocity=5.0):
        """Move UAV to target position."""
        if not self.client:
            return

        airsim_pos = self.sim_to_airsim_pos(target_pos)
        self.client.moveToPositionAsync(
            airsim_pos.x_val, airsim_pos.y_val, airsim_pos.z_val,
            velocity
        ).join()

    def get_camera_image(self):
        """Get camera image from AirSim."""
        if not self.client:
            # Return placeholder image
            img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            cv2.putText(img, "AirSim Not Connected", (100, 540),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            return img

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        if responses and responses[0].width > 0:
            img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img = img.reshape(responses[0].height, responses[0].width, 3)
            img = cv2.resize(img, self.resolution)
            return img

        return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

    def add_overlay(self, frame, scenario, method_name, throughput, position):
        """Add information overlay to frame."""
        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "MI-RL UAV Relay Positioning", (30, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Method info
        color = (0, 255, 0) if 'MI-RL' in method_name else (0, 165, 255)
        cv2.putText(frame, f"Method: {method_name}", (30, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Throughput
        cv2.putText(frame, f"Throughput: {throughput:.1f} Mbps", (30, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Position
        cv2.putText(frame, f"Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})m",
                   (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Thesis statement at bottom
        cv2.putText(frame, '"A little bit of Math + a little bit of RL"',
                   (self.resolution[0]//2 - 300, self.resolution[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame

    def run_demo(self, output_path='mi_rl_airsim_demo.mp4', duration=60):
        """Run the full AirSim demonstration."""
        print(f"\n{'='*60}")
        print("MI-RL AirSim Demo")
        print(f"{'='*60}")
        print(f"Duration: {duration}s, FPS: {self.fps}")
        print(f"Output: {output_path}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, self.fps, self.resolution
        )

        total_frames = duration * self.fps
        frames_per_scenario = total_frames // len(self.scenarios)

        # Takeoff
        if self.client:
            print("\nTaking off...")
            self.client.takeoffAsync().join()
            time.sleep(2)

        start_time = time.time()

        for frame_idx in range(total_frames):
            # Determine current scenario and phase
            scenario_idx = frame_idx // frames_per_scenario
            scenario_idx = min(scenario_idx, len(self.scenarios) - 1)
            scenario = self.scenarios[scenario_idx]

            local_frame = frame_idx % frames_per_scenario
            phase_duration = frames_per_scenario // 3

            # Compute positions
            sca_pos, _ = self.sca_solver.solve(scenario, verbose=False)
            mirl_pos = self.agent.get_position(scenario, deterministic=True)

            # Determine current method and position
            if local_frame < phase_duration:
                # Phase 1: Show SCA position
                current_pos = sca_pos
                method_name = "SCA-20 (Classical)"
            elif local_frame < 2 * phase_duration:
                # Phase 2: Transition to MI-RL
                t = (local_frame - phase_duration) / phase_duration
                current_pos = sca_pos + t * (mirl_pos - sca_pos)
                method_name = "Transitioning to MI-RL..."
            else:
                # Phase 3: Show MI-RL position
                current_pos = mirl_pos
                method_name = "MI-RL (Math + RL)"

            # Move UAV
            if frame_idx % self.fps == 0:  # Update position every second
                self.move_to_position(current_pos)

            # Compute throughput at current position
            metrics = compute_channel_metrics(current_pos, scenario)
            throughput = metrics['total_throughput']

            # Capture frame
            frame = self.get_camera_image()

            # Add overlay
            frame = self.add_overlay(frame, scenario, method_name, throughput, current_pos)

            # Write frame
            self.video_writer.write(frame)

            # Progress
            if frame_idx % (self.fps * 5) == 0:
                elapsed = time.time() - start_time
                print(f"  Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%), "
                      f"elapsed: {elapsed:.1f}s")

        # Cleanup
        self.video_writer.release()

        if self.client:
            print("\nLanding...")
            self.client.landAsync().join()
            self.client.armDisarm(False)

        print(f"\n✅ Demo video saved: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='MI-RL AirSim Demo')
    parser.add_argument('--checkpoint', type=str, default='latest',
                        help='Path to MI-RL checkpoint')
    parser.add_argument('--duration', type=int, default=60,
                        help='Demo duration in seconds')
    parser.add_argument('--output', type=str, default='mi_rl_airsim_demo.mp4',
                        help='Output video filename')
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.checkpoint == 'latest':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(
            script_dir, '../../results/mi_rl/checkpoints/checkpoint_latest.pt'
        )
    else:
        checkpoint_path = args.checkpoint

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../../videos')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output)

    # Run demo
    demo = MIRLAirSimDemo(checkpoint_path=checkpoint_path)
    demo.run_demo(output_path=output_path, duration=args.duration)


if __name__ == "__main__":
    main()
