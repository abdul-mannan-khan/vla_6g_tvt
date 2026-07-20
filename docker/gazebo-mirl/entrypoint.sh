#!/bin/bash
set -e

# Check if running in interactive mode (with supervisor)
if [[ "$1" == "supervisord" ]]; then
    echo "=============================================="
    echo "MI-RL Demo - Interactive Remote Desktop Mode"
    echo "=============================================="
    echo ""
    echo "Starting VNC server and noVNC web interface..."
    echo ""
    echo "Connect via:"
    echo "  - Browser (noVNC): http://<host>:6080/vnc.html"
    echo "  - VNC Client: <host>:5901 (password: mirl2024)"
    echo ""
    echo "Once connected, open a terminal and run:"
    echo "  gz sim /workspace/worlds/mirl_demo.sdf"
    echo ""
    echo "=============================================="
    exec "$@"
else
    # Headless mode for rendering
    echo "Starting Xvfb on display :99..."
    rm -f /tmp/.X99-lock
    Xvfb :99 -screen 0 1920x1080x24 +extension GLX +render -noreset &
    XVFB_PID=$!
    sleep 2

    # Verify Xvfb is running
    if ! kill -0 $XVFB_PID 2>/dev/null; then
        echo "ERROR: Xvfb failed to start"
        exit 1
    fi

    export DISPLAY=:99

    # Test OpenGL
    echo "Testing OpenGL rendering..."
    if command -v glxinfo &> /dev/null; then
        glxinfo | grep "OpenGL renderer" || echo "glxinfo not available, continuing..."
    fi

    echo "Gazebo MIRL Demo environment ready (headless mode)"
    echo "Worlds directory: /workspace/worlds"
    echo "Videos will be saved to: /workspace/videos"
    echo ""

    # Execute the command passed to docker run
    exec "$@"
fi
