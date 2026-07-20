#!/bin/bash
# Record MI-RL UAV Relay Demo Video from Gazebo Harmonic
# Usage: ./record_demo.sh [duration_seconds] [output_name]

DURATION=${1:-60}
OUTPUT_NAME=${2:-mirl_gazebo_demo}
OUTPUT_DIR=/workspace/videos
WORLD_FILE=/workspace/worlds/urban_mirl_demo.sdf

echo "=== MI-RL Gazebo Demo Recorder ==="
echo "Duration: ${DURATION}s"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}.mp4"
echo ""

# Ensure display is set
export DISPLAY=:99

# Start Gazebo in background with rendering
echo "Starting Gazebo simulation..."
gz sim ${WORLD_FILE} -r --headless-rendering &
GZ_PID=$!
sleep 5

# Check if Gazebo started
if ! kill -0 $GZ_PID 2>/dev/null; then
    echo "ERROR: Gazebo failed to start"
    exit 1
fi

echo "Gazebo started with PID: $GZ_PID"

# Start ffmpeg screen recording
echo "Starting video recording..."
ffmpeg -y \
    -f x11grab \
    -video_size 1920x1080 \
    -framerate 30 \
    -i :99.0 \
    -t ${DURATION} \
    -c:v libx264 \
    -preset fast \
    -crf 20 \
    -pix_fmt yuv420p \
    ${OUTPUT_DIR}/${OUTPUT_NAME}_raw.mp4 &
FFMPEG_PID=$!

echo "Recording for ${DURATION} seconds..."
sleep ${DURATION}

# Stop recording
kill $FFMPEG_PID 2>/dev/null || true
sleep 2

# Stop Gazebo
echo "Stopping Gazebo..."
kill $GZ_PID 2>/dev/null || true
sleep 2

# Verify and finalize video
if [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}_raw.mp4" ]; then
    echo "Finalizing video encoding..."
    ffmpeg -y \
        -i ${OUTPUT_DIR}/${OUTPUT_NAME}_raw.mp4 \
        -c:v libx264 \
        -preset slow \
        -crf 18 \
        -pix_fmt yuv420p \
        -movflags +faststart \
        ${OUTPUT_DIR}/${OUTPUT_NAME}.mp4

    rm -f ${OUTPUT_DIR}/${OUTPUT_NAME}_raw.mp4

    echo ""
    echo "=== Recording Complete ==="
    ls -lh ${OUTPUT_DIR}/${OUTPUT_NAME}.mp4
else
    echo "ERROR: Recording failed - no video file produced"
    exit 1
fi
