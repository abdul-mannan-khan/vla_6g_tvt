#!/bin/bash
# Build and push Gazebo MI-RL Demo Docker image

IMAGE_NAME="abdulmannan617/gazebo-mirl-demo"
TAG="${1:-latest}"

echo "=== Building Gazebo MI-RL Demo Image ==="
echo "Image: ${IMAGE_NAME}:${TAG}"
echo ""

cd "$(dirname "$0")"

# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*.py
chmod +x entrypoint.sh

# Build image
docker build -t ${IMAGE_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""

    # Push to Docker Hub
    echo "Pushing to Docker Hub..."
    docker push ${IMAGE_NAME}:${TAG}

    if [ $? -eq 0 ]; then
        echo ""
        echo "=== Push Complete ==="
        echo "Image available at: ${IMAGE_NAME}:${TAG}"
    else
        echo "Push failed. Login with: docker login"
    fi
else
    echo "Build failed!"
    exit 1
fi
