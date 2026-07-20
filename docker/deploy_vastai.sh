#!/bin/bash
# =============================================================================
# Deploy MI-RL Docker Container to Vast.ai
# =============================================================================
#
# Usage:
#   ./deploy_vastai.sh [build|push|deploy|run|all]
#
# Prerequisites:
#   - Docker installed and running
#   - Vast.ai CLI configured (vastai set api-key YOUR_KEY)
#   - Docker Hub account (for pushing)
# =============================================================================

set -e

DOCKER_IMAGE="mirl-6g"
DOCKER_TAG="latest"
DOCKER_REGISTRY=""  # Set to your Docker Hub username, e.g., "username/mirl-6g"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[MI-RL]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

build_image() {
    log "Building Docker image..."
    cd "$PROJECT_DIR"
    docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} -f docker/Dockerfile.mirl .
    success "Image built: ${DOCKER_IMAGE}:${DOCKER_TAG}"
}

push_image() {
    if [ -z "$DOCKER_REGISTRY" ]; then
        log "DOCKER_REGISTRY not set. Set it to push to Docker Hub."
        return 1
    fi
    log "Pushing to Docker Hub..."
    docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_REGISTRY}:${DOCKER_TAG}
    docker push ${DOCKER_REGISTRY}:${DOCKER_TAG}
    success "Pushed: ${DOCKER_REGISTRY}:${DOCKER_TAG}"
}

deploy_vastai() {
    log "Searching for GPU instance..."

    OFFER_ID=$(vastai search offers 'gpu_ram >= 8 dph < 0.15 reliability > 0.95' --raw 2>/dev/null | \
        python3 -c "import json,sys; o=json.load(sys.stdin); print(min([x for x in o if 'RTX' in x.get('gpu_name','')], key=lambda x: x['dph_total'])['id'] if o else '')" 2>/dev/null)

    if [ -z "$OFFER_ID" ]; then
        log "No suitable offers found"
        return 1
    fi

    log "Creating instance with offer $OFFER_ID..."

    IMAGE="${DOCKER_REGISTRY:-pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime}"

    RESULT=$(vastai create instance $OFFER_ID \
        --image "$IMAGE" \
        --disk 50 \
        --ssh \
        --label "mirl-$(date +%Y%m%d%H%M)" 2>&1)

    INSTANCE_ID=$(echo "$RESULT" | grep -oP "'new_contract': \K[0-9]+" || echo "")

    if [ -n "$INSTANCE_ID" ]; then
        success "Instance created: $INSTANCE_ID"
        echo "  Monitor: vastai show instance $INSTANCE_ID"
        echo "  Logs:    vastai logs $INSTANCE_ID"
        echo "  Destroy: vastai destroy instance $INSTANCE_ID"
    else
        log "Failed to create instance: $RESULT"
    fi
}

run_local() {
    log "Running locally with Docker..."
    docker run --gpus all -it --rm \
        -v "$PROJECT_DIR/results:/workspace/results" \
        ${DOCKER_IMAGE}:${DOCKER_TAG} \
        python3 scripts/algorithm_comparison.py --scenarios 50 --episodes 300
}

case "${1:-all}" in
    build)  build_image ;;
    push)   push_image ;;
    deploy) deploy_vastai ;;
    run)    run_local ;;
    all)
        build_image
        push_image
        deploy_vastai
        ;;
    *)
        echo "Usage: $0 [build|push|deploy|run|all]"
        exit 1
        ;;
esac
