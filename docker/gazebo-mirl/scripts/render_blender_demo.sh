#!/bin/bash
# Render MI-RL demo video using Blender
# Usage: ./render_blender_demo.sh [output_name]

OUTPUT_NAME=${1:-mi_rl_blender_demo}
OUTPUT_DIR=/workspace/videos

echo "=== MI-RL Blender Demo Renderer ==="
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}.mp4"
echo ""

# Ensure Xvfb is running
if ! pgrep Xvfb > /dev/null; then
    echo "Starting Xvfb..."
    Xvfb :99 -screen 0 1920x1080x24 +extension GLX +render -noreset &
    sleep 2
fi
export DISPLAY=:99

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Update output path in script
sed -i "s|OUTPUT_PATH = .*|OUTPUT_PATH = \"${OUTPUT_DIR}/${OUTPUT_NAME}.mp4\"|" /workspace/scripts/blender_mirl_demo_fast.py

# Run Blender
echo "Rendering..."
blender --background --python /workspace/scripts/blender_mirl_demo_fast.py

# Verify
if [ -f "${OUTPUT_DIR}/${OUTPUT_NAME}.mp4" ]; then
    echo ""
    echo "=== Complete ==="
    ls -lh ${OUTPUT_DIR}/${OUTPUT_NAME}.mp4
else
    echo "ERROR: Rendering failed"
    exit 1
fi
