#!/bin/bash
# Deploy SGAC Training to Vast.ai GPU Instance
# Usage: ./deploy_vastai.sh [episodes] [gpu_type]

set -e

# Configuration
EPISODES=${1:-2000}
GPU_TYPE=${2:-"RTX_3090"}
MAX_PRICE=${3:-0.50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "SGAC Training - Vast.ai Deployment"
echo "========================================"
echo "Episodes: $EPISODES"
echo "GPU Type: $GPU_TYPE"
echo "Max Price: \$$MAX_PRICE/hr"
echo ""

# Check for API key
if [ -f "$PROJECT_ROOT/.secrets/vastai_api_key" ]; then
    export VAST_API_KEY=$(cat "$PROJECT_ROOT/.secrets/vastai_api_key")
    echo "Using API key from .secrets/vastai_api_key"
elif [ -f ~/.config/vastai/vast_api_key ]; then
    export VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key)
    echo "Using API key from ~/.config/vastai/vast_api_key"
else
    echo "Error: Vast.ai API key not found"
    echo "Please set up your API key:"
    echo "  mkdir -p ~/.config/vastai"
    echo "  echo 'YOUR_API_KEY' > ~/.config/vastai/vast_api_key"
    exit 1
fi

# Verify vastai CLI
if ! command -v vastai &> /dev/null; then
    echo "Installing vastai CLI..."
    pip install vastai --upgrade
fi

echo ""
echo "Searching for $GPU_TYPE instance..."

# Search for GPU instance
SEARCH_QUERY="gpu_name=$GPU_TYPE num_gpus=1 dph<$MAX_PRICE inet_down>100 cuda_vers>=12.0 reliability>0.95 disk_space>=30"
INSTANCE=$(vastai search offers "$SEARCH_QUERY" --order 'dph-' --limit 1 --raw 2>/dev/null)

if [ -z "$INSTANCE" ] || [ "$INSTANCE" == "[]" ]; then
    echo "No $GPU_TYPE found. Trying RTX_4090..."
    GPU_TYPE="RTX_4090"
    SEARCH_QUERY="gpu_name=$GPU_TYPE num_gpus=1 dph<1.0 inet_down>100 cuda_vers>=12.0 reliability>0.95 disk_space>=30"
    INSTANCE=$(vastai search offers "$SEARCH_QUERY" --order 'dph-' --limit 1 --raw 2>/dev/null)
fi

if [ -z "$INSTANCE" ] || [ "$INSTANCE" == "[]" ]; then
    echo "No RTX_4090 found. Trying any RTX 30/40 series..."
    SEARCH_QUERY="gpu_ram>=10 num_gpus=1 dph<1.0 inet_down>100 cuda_vers>=12.0 reliability>0.90 disk_space>=30"
    INSTANCE=$(vastai search offers "$SEARCH_QUERY" --order 'dph-' --limit 1 --raw 2>/dev/null)
fi

if [ -z "$INSTANCE" ] || [ "$INSTANCE" == "[]" ]; then
    echo "Error: No suitable GPU instance found"
    echo "Try increasing MAX_PRICE or waiting for availability"
    exit 1
fi

INSTANCE_ID=$(echo "$INSTANCE" | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")
INSTANCE_GPU=$(echo "$INSTANCE" | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['gpu_name'])")
INSTANCE_PRICE=$(echo "$INSTANCE" | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['dph_total'])")

echo "Found instance: ID=$INSTANCE_ID, GPU=$INSTANCE_GPU, Price=\$$INSTANCE_PRICE/hr"

# Create startup script
STARTUP_SCRIPT='
#!/bin/bash
set -e
echo "========================================="
echo "SGAC Training Starting..."
echo "========================================="

# Install dependencies
pip install torch numpy scipy matplotlib tqdm --upgrade

# Clone or update repository
cd /workspace
if [ -d "vla_6g_tvt" ]; then
    cd vla_6g_tvt && git pull
else
    git clone https://github.com/abdul-mannan-khan/vla_6g_tvt.git
    cd vla_6g_tvt
fi

# Run training
cd sgac_pytorch
python train.py \
    --episodes '$EPISODES' \
    --eval-interval 100 \
    --save-interval 500 \
    --num-scenarios 100 \
    --num-eval-scenarios 50 \
    --output-dir /workspace/results \
    --verbose

echo "========================================="
echo "Training Complete!"
echo "Results saved to /workspace/results"
echo "========================================="
'

echo ""
echo "Creating instance..."
CREATE_OUTPUT=$(vastai create instance $INSTANCE_ID \
    --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    --disk 30 \
    --onstart-cmd "$STARTUP_SCRIPT" \
    2>&1)

echo "$CREATE_OUTPUT"

# Extract instance ID from create output
NEW_INSTANCE_ID=$(echo "$CREATE_OUTPUT" | grep -oP 'new instance ID is \K\d+' || echo "")

if [ -z "$NEW_INSTANCE_ID" ]; then
    NEW_INSTANCE_ID=$(echo "$CREATE_OUTPUT" | grep -oP '\d+' | head -1)
fi

echo ""
echo "========================================"
echo "Instance Created!"
echo "========================================"
echo ""
echo "Instance ID: $NEW_INSTANCE_ID"
echo "GPU: $INSTANCE_GPU"
echo "Price: \$$INSTANCE_PRICE/hr"
echo ""
echo "Commands:"
echo "  View status:  vastai show instances"
echo "  View logs:    vastai logs $NEW_INSTANCE_ID"
echo "  SSH connect:  vastai ssh $NEW_INSTANCE_ID"
echo "  Stop:         vastai stop instance $NEW_INSTANCE_ID"
echo "  Destroy:      vastai destroy instance $NEW_INSTANCE_ID"
echo ""
echo "When training completes, download results:"
echo "  vastai copy $NEW_INSTANCE_ID:/workspace/results ./results_vastai/"
echo ""

# Save instance info
echo "$NEW_INSTANCE_ID" > "$SCRIPT_DIR/.current_instance"
echo "Instance ID saved to .current_instance"

# Wait for instance to start and show initial logs
echo "Waiting for instance to start (60 seconds)..."
sleep 60

echo ""
echo "Initial logs:"
vastai logs $NEW_INSTANCE_ID --tail 50 2>/dev/null || echo "(Logs not yet available)"
