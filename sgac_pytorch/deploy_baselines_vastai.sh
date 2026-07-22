#!/bin/bash
# Deploy baseline training to Vast.ai
# Usage: ./deploy_baselines_vastai.sh <instance_id>

INSTANCE_ID=${1:-""}

if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: $0 <instance_id>"
    echo ""
    echo "To find available instances:"
    echo "  vastai search offers 'gpu_name=RTX_4090 num_gpus=1 dph<0.5'"
    echo ""
    echo "To create an instance:"
    echo "  vastai create instance <offer_id> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime --disk 20"
    exit 1
fi

# Get instance info
echo "Getting instance info..."
INSTANCE_INFO=$(vastai show instance $INSTANCE_ID --raw)
SSH_HOST=$(echo $INSTANCE_INFO | jq -r '.ssh_host')
SSH_PORT=$(echo $INSTANCE_INFO | jq -r '.ssh_port')

echo "Instance: $INSTANCE_ID"
echo "SSH: $SSH_HOST:$SSH_PORT"

# Create workspace on remote
echo "Setting up remote workspace..."
ssh -p $SSH_PORT root@$SSH_HOST "mkdir -p /workspace/sgac_pytorch/baselines"

# Copy files
echo "Copying baseline agents..."
scp -P $SSH_PORT \
    baselines/ddpg_agent.py \
    baselines/td3_agent.py \
    baselines/sac_agent.py \
    baselines/ppo_agent.py \
    baselines/train_baselines.py \
    baselines/__init__.py \
    root@$SSH_HOST:/workspace/sgac_pytorch/baselines/

echo "Copying environment and dependencies..."
scp -P $SSH_PORT \
    environment.py \
    networks.py \
    root@$SSH_HOST:/workspace/sgac_pytorch/

# Install dependencies and run training
echo "Starting training..."
ssh -p $SSH_PORT root@$SSH_HOST << 'EOF'
cd /workspace/sgac_pytorch

# Install dependencies
pip install numpy torch --quiet

# Run all baselines
echo "Training all baselines (DDPG, TD3, SAC, PPO)..."
cd baselines
python train_baselines.py --agent all --episodes 2000 --output_dir /workspace/results_baselines

echo "Training complete!"
ls -la /workspace/results_baselines/
EOF

echo ""
echo "Training started. To check progress:"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/results_baselines/*.log'"
echo ""
echo "To download results when done:"
echo "  scp -P $SSH_PORT -r root@$SSH_HOST:/workspace/results_baselines ./results_baselines_vastai"
