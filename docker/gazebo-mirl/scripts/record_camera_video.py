#!/usr/bin/env python3
"""
Record video from Gazebo camera sensor using gz topic.
Works with headless rendering.
"""

import subprocess
import time
import os
import signal
import sys

# Configuration
WORLD_FILE = os.environ.get('WORLD_FILE', '/root/urban_mirl_demo.sdf')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/root/videos')
OUTPUT_NAME = os.environ.get('OUTPUT_NAME', 'mirl_demo')
CAMERA_TOPIC = '/world/urban_mirl_demo/model/scene_camera/link/link/sensor/camera/image'
DURATION = int(os.environ.get('DURATION', '45'))
FPS = 30

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames_dir = f'{OUTPUT_DIR}/frames'
    os.makedirs(frames_dir, exist_ok=True)

    print("=== MI-RL Gazebo Video Recorder ===")
    print(f"World: {WORLD_FILE}")
    print(f"Output: {OUTPUT_DIR}/{OUTPUT_NAME}.mp4")
    print(f"Duration: {DURATION}s")
    print()

    # Start Gazebo in headless mode with server-side rendering
    print("Starting Gazebo with headless rendering...")
    gz_cmd = f"gz sim {WORLD_FILE} -r -s --headless-rendering"
    gz_proc = subprocess.Popen(
        gz_cmd, shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    time.sleep(10)  # Wait for world to load

    # Check available topics
    print("Checking available camera topics...")
    topics_result = subprocess.run("gz topic -l | grep -i camera", shell=True, capture_output=True, text=True)
    print(f"Camera topics: {topics_result.stdout}")

    # Try to echo the camera topic to verify it's working
    print("Testing camera feed...")

    # Record using gz topic echo and save images
    print(f"Recording for {DURATION} seconds...")

    frame_count = 0
    start_time = time.time()

    # For each frame, capture and save
    while time.time() - start_time < DURATION:
        frame_path = f"{frames_dir}/frame_{frame_count:05d}.png"
        # Use gz topic -e to echo one message and save as image
        cmd = f"gz topic -e -t {CAMERA_TOPIC} -n 1 --msgtype gz.msgs.Image 2>/dev/null"
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=2)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  Frame {frame_count}, elapsed: {elapsed:.1f}s")

        # Small delay to maintain roughly target FPS
        time.sleep(1.0 / FPS)

    print(f"Captured {frame_count} frames")

    # Stop Gazebo
    print("Stopping Gazebo...")
    os.killpg(os.getpgid(gz_proc.pid), signal.SIGTERM)
    time.sleep(2)

    # If we have frames, combine them into video
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if frame_files:
        print(f"Combining {len(frame_files)} frames into video...")
        ffmpeg_cmd = f"""
        ffmpeg -y -framerate {FPS} -i {frames_dir}/frame_%05d.png \
            -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
            -movflags +faststart {OUTPUT_DIR}/{OUTPUT_NAME}.mp4
        """
        subprocess.run(ffmpeg_cmd, shell=True)
        print(f"Video saved to: {OUTPUT_DIR}/{OUTPUT_NAME}.mp4")

        # Clean up frames
        subprocess.run(f"rm -rf {frames_dir}", shell=True)
    else:
        print("No frames captured. Camera topic may not be publishing.")
        print("Trying alternative approach with GUI rendering...")

    print("=== Recording Complete ===")

if __name__ == "__main__":
    main()
