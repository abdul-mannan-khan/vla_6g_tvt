#!/bin/bash
# Full MI-RL Demo Recording with Animation
# Usage: ./record_full_demo.sh [output_name]

OUTPUT_NAME=${1:-mirl_uav_relay_demo}
OUTPUT_DIR=/workspace/videos
WORLD_FILE=/workspace/worlds/urban_mirl_demo.sdf
SCRIPT_DIR="$(dirname "$0")"

echo "=== MI-RL Full Demo Recorder ==="
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}.mp4"
echo ""

# Ensure display
export DISPLAY=:99

# Clean up any previous Gazebo instances
pkill -9 gz 2>/dev/null || true
pkill -9 ruby 2>/dev/null || true
sleep 2

# Start Gazebo with GUI for visual recording
echo "Starting Gazebo simulation with GUI..."
gz sim ${WORLD_FILE} -r &
GZ_PID=$!
sleep 8  # Give Gazebo time to load world and assets

# Check if Gazebo started
if ! kill -0 $GZ_PID 2>/dev/null; then
    echo "ERROR: Gazebo failed to start"
    exit 1
fi
echo "Gazebo running (PID: $GZ_PID)"

# Start screen recording
echo "Starting video capture..."
ffmpeg -y \
    -f x11grab \
    -video_size 1920x1080 \
    -framerate 30 \
    -i :99.0 \
    -c:v libx264 \
    -preset ultrafast \
    -crf 18 \
    ${OUTPUT_DIR}/${OUTPUT_NAME}_raw.mp4 &
FFMPEG_PID=$!
sleep 2

echo "Recording started (PID: $FFMPEG_PID)"

# Run animation script
echo "Running MI-RL animation..."
python3 ${SCRIPT_DIR}/animate_mirl_demo.py

# Extra hold time at end
sleep 3

# Stop recording
echo "Stopping recording..."
kill $FFMPEG_PID 2>/dev/null || true
sleep 2

# Stop Gazebo
echo "Stopping Gazebo..."
kill $GZ_PID 2>/dev/null || true
pkill -9 gz 2>/dev/null || true
sleep 2

# Finalize video with high quality encoding
if [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}_raw.mp4" ]; then
    echo "Encoding final video..."
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
    echo ""
    echo "Video saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}.mp4"
else
    echo "ERROR: Recording failed"
    exit 1
fi
