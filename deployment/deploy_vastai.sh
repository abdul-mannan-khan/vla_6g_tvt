#!/bin/bash
# Deploy Math-Informed RL to Vast.ai

set -e

# Configuration
EPISODES=${1:-2000}
EVAL_INTERVAL=${2:-100}

echo "========================================"
echo "Math-Informed RL - Vast.ai Deployment"
echo "========================================"
echo "Episodes: $EPISODES"
echo "Eval interval: $EVAL_INTERVAL"
echo ""

# Check for API key
if [ ! -f ~/.config/vastai/vast_api_key ] && [ ! -f ../.secrets/vastai_api_key ]; then
    echo "Error: Vast.ai API key not found"
    echo "Please set up your API key:"
    echo "  mkdir -p ~/.config/vastai"
    echo "  echo 'YOUR_API_KEY' > ~/.config/vastai/vast_api_key"
    exit 1
fi

# Use local key if available
if [ -f ../.secrets/vastai_api_key ]; then
    export VAST_API_KEY=$(cat ../.secrets/vastai_api_key)
fi

# Search for GPU instance
echo "Searching for GPU instance..."
INSTANCE=$(vastai search offers \
    'gpu_name=RTX_3090 num_gpus=1 dph<0.5 inet_down>100 cuda_vers>=12.0 reliability>0.95' \
    --order 'dph-' \
    --limit 1 \
    --raw)

if [ -z "$INSTANCE" ]; then
    echo "No suitable instance found. Trying RTX 4090..."
    INSTANCE=$(vastai search offers \
        'gpu_name=RTX_4090 num_gpus=1 dph<1.0 inet_down>100 cuda_vers>=12.0 reliability>0.95' \
        --order 'dph-' \
        --limit 1 \
        --raw)
fi

if [ -z "$INSTANCE" ]; then
    echo "Error: No suitable GPU instance found"
    exit 1
fi

INSTANCE_ID=$(echo "$INSTANCE" | jq -r '.[0].id')
echo "Found instance: $INSTANCE_ID"

# Create instance
echo "Creating instance..."
vastai create instance $INSTANCE_ID \
    --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    --disk 20 \
    --onstart-cmd "pip install scipy numpy matplotlib tqdm && cd /workspace && git clone https://github.com/abdul-mannan-khan/vla_6g_tvt.git && cd vla_6g_tvt && python scripts/mi_rl/train_mi_rl.py --episodes $EPISODES --eval-interval $EVAL_INTERVAL"

echo ""
echo "Instance created. Use 'vastai show instances' to monitor."
echo "Use 'vastai logs <instance_id>' to view training output."
echo ""
echo "When done, download results with:"
echo "  vastai copy <instance_id>:/workspace/vla_6g_tvt/results/mi_rl/ ./results/"
