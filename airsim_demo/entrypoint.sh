#!/bin/bash
# MI-RL 6G UAV Relay Demo - Entry Point

set -e

echo "=========================================="
echo "MI-RL 6G UAV Relay - AirSim Demo"
echo "=========================================="

# Start virtual display
echo "Starting virtual display..."
Xvfb :99 -screen 0 ${RESOLUTION:-1920x1080x24} &
sleep 2

# Start window manager
echo "Starting window manager..."
fluxbox &
sleep 1

# Start VNC server
echo "Starting VNC server..."
x11vnc -display :99 -forever -shared -rfbport 5900 &
sleep 1

# Start noVNC
echo "Starting noVNC web interface..."
websockify --web=/usr/share/novnc/ 8080 localhost:5900 &
sleep 1

echo ""
echo "=========================================="
echo "Services Started:"
echo "  - VNC: localhost:5900"
echo "  - Web: http://localhost:8080/vnc.html"
echo "=========================================="
echo ""

# Handle command
case "$1" in
    "demo")
        echo "Running MI-RL demo..."
        cd /workspace
        export PYTHONPATH=/workspace:$PYTHONPATH
        python3 airsim_demo/mi_rl_demo_recorder.py --duration 60 --output /workspace/videos/mi_rl_demo.mp4
        ;;
    "interactive")
        echo "Starting interactive mode..."
        exec /bin/bash
        ;;
    "record")
        echo "Recording demo video..."
        cd /workspace
        export PYTHONPATH=/workspace:$PYTHONPATH
        python3 airsim_demo/mi_rl_demo_recorder.py --duration ${2:-60} --output /workspace/videos/mi_rl_demo.mp4
        ;;
    *)
        echo "Usage: docker run <image> [demo|interactive|record <duration>]"
        exec "$@"
        ;;
esac

echo ""
echo "Demo complete! Video saved to /workspace/videos/"
echo "Access at: http://localhost:8080/vnc.html"

# Keep container running
tail -f /dev/null
