#!/usr/bin/env python3
"""
MI-RL UAV Relay Positioning Demo Animation
Animates UAV movement showing optimization process in Gazebo Harmonic.
"""

import subprocess
import time
import math
import sys

# Demo configuration
BS_POS = (0, 0, 25)  # Base station position
USER_POSITIONS = [
    (30, 20, 0),
    (-25, 35, 0),
    (40, -15, 0),
    (-30, -25, 0),
    (15, 45, 0),
]

# UAV flight parameters
UAV_HEIGHT = 35.0
OPTIMIZATION_STEPS = 50
DEMO_DURATION = 45  # seconds

def set_model_pose(model_name, x, y, z, roll=0, pitch=0, yaw=0):
    """Set model pose using gz service."""
    pose_msg = f"""
    {{
        "name": "{model_name}",
        "pose": {{
            "position": {{"x": {x}, "y": {y}, "z": {z}}},
            "orientation": {{
                "x": {math.sin(roll/2)*math.cos(pitch/2)*math.cos(yaw/2) - math.cos(roll/2)*math.sin(pitch/2)*math.sin(yaw/2)},
                "y": {math.cos(roll/2)*math.sin(pitch/2)*math.cos(yaw/2) + math.sin(roll/2)*math.cos(pitch/2)*math.sin(yaw/2)},
                "z": {math.cos(roll/2)*math.cos(pitch/2)*math.sin(yaw/2) - math.sin(roll/2)*math.sin(pitch/2)*math.cos(yaw/2)},
                "w": {math.cos(roll/2)*math.cos(pitch/2)*math.cos(yaw/2) + math.sin(roll/2)*math.sin(pitch/2)*math.sin(yaw/2)}
            }}
        }}
    }}
    """
    cmd = f'gz service -s /world/urban_mirl_demo/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1000 --req \'{pose_msg.strip()}\''
    subprocess.run(cmd, shell=True, capture_output=True)

def calculate_optimal_position(user_positions, bs_pos):
    """Calculate optimal UAV relay position using geometric center weighted by channel."""
    # Simple weighted centroid for demo (actual MI-RL uses learned optimization)
    total_weight = 0
    weighted_x, weighted_y = 0, 0

    for ux, uy, uz in user_positions:
        # Distance-based weight (closer users get more weight)
        dist_to_bs = math.sqrt((ux - bs_pos[0])**2 + (uy - bs_pos[1])**2)
        weight = 1.0 / (1.0 + dist_to_bs / 50.0)
        weighted_x += ux * weight
        weighted_y += uy * weight
        total_weight += weight

    opt_x = weighted_x / total_weight
    opt_y = weighted_y / total_weight
    return opt_x, opt_y

def animate_optimization(start_pos, end_pos, steps, step_delay=0.1):
    """Animate UAV moving from start to optimized position with exploration."""
    print(f"Animating optimization: {start_pos} -> {end_pos}")

    sx, sy = start_pos
    ex, ey = end_pos

    for i in range(steps):
        t = i / (steps - 1)

        # Add exploration noise that decreases over time (simulating convergence)
        exploration = (1 - t) * 5.0
        noise_x = exploration * math.sin(i * 0.5)
        noise_y = exploration * math.cos(i * 0.7)

        # Interpolate with easing
        ease_t = t * t * (3 - 2 * t)  # Smooth step
        x = sx + (ex - sx) * ease_t + noise_x
        y = sy + (ey - sy) * ease_t + noise_y

        set_model_pose("x3_uav", x, y, UAV_HEIGHT)
        time.sleep(step_delay)

    # Final position exactly at optimum
    set_model_pose("x3_uav", ex, ey, UAV_HEIGHT)

def main():
    print("=== MI-RL UAV Relay Positioning Demo ===")
    print(f"Base Station: {BS_POS}")
    print(f"Users: {len(USER_POSITIONS)}")
    print("")

    # Calculate optimal position
    opt_x, opt_y = calculate_optimal_position(USER_POSITIONS, BS_POS)
    print(f"Optimal UAV position: ({opt_x:.1f}, {opt_y:.1f}, {UAV_HEIGHT})")

    # Phase 1: Initial random position
    print("\n[Phase 1] Initial Position")
    init_x, init_y = -40, -40
    set_model_pose("x3_uav", init_x, init_y, UAV_HEIGHT)
    time.sleep(3)

    # Phase 2: Exploration phase (random search)
    print("\n[Phase 2] Random Exploration")
    random_positions = [
        (20, -30), (-35, 25), (45, 30), (-20, -45), (35, -10)
    ]
    for rx, ry in random_positions:
        set_model_pose("x3_uav", rx, ry, UAV_HEIGHT)
        time.sleep(1.5)

    # Phase 3: MI-RL Optimization
    print("\n[Phase 3] MI-RL Optimization")
    current_x, current_y = random_positions[-1]
    animate_optimization(
        (current_x, current_y),
        (opt_x, opt_y),
        OPTIMIZATION_STEPS,
        step_delay=0.15
    )

    # Phase 4: Hold at optimal position
    print("\n[Phase 4] Optimal Position Reached")
    time.sleep(5)

    # Phase 5: Dynamic adaptation (users move slightly)
    print("\n[Phase 5] Dynamic Adaptation")
    for _ in range(3):
        # Simulate small adjustments
        adj_x = opt_x + 3 * math.sin(time.time())
        adj_y = opt_y + 3 * math.cos(time.time())
        set_model_pose("x3_uav", adj_x, adj_y, UAV_HEIGHT)
        time.sleep(2)

    # Return to optimal
    set_model_pose("x3_uav", opt_x, opt_y, UAV_HEIGHT)

    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
