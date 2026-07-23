#!/bin/bash
# Deploy FIXED SGAC training to Vast.ai

set -e

echo "=============================================="
echo "FIXED SGAC Deployment to Vast.ai"
echo "=============================================="

# Check for vastai CLI
if ! command -v vastai &> /dev/null; then
    echo "Error: vastai CLI not found"
    exit 1
fi

# Search for a reliable RTX 4090
echo "Searching for GPU instance..."
OFFER_ID=$(vastai search offers 'gpu_name=RTX_4090 num_gpus=1 dph<0.5 inet_down>100 reliability>0.95' --raw 2>/dev/null | \
    jq -r 'sort_by(.dph_total) | .[0].id' 2>/dev/null)

if [ -z "$OFFER_ID" ] || [ "$OFFER_ID" = "null" ]; then
    echo "No suitable offers found. Trying with relaxed constraints..."
    OFFER_ID=$(vastai search offers 'gpu_name=RTX_4090 num_gpus=1 dph<0.6' --raw 2>/dev/null | \
        jq -r 'sort_by(.dph_total) | .[0].id' 2>/dev/null)
fi

if [ -z "$OFFER_ID" ] || [ "$OFFER_ID" = "null" ]; then
    echo "Error: No GPU offers found"
    exit 1
fi

echo "Found offer: $OFFER_ID"

# Create instance
echo "Creating instance..."
RESULT=$(vastai create instance $OFFER_ID --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime --disk 20 --raw 2>&1)
INSTANCE_ID=$(echo $RESULT | jq -r '.new_contract')

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "null" ]; then
    echo "Error creating instance: $RESULT"
    exit 1
fi

echo "Instance created: $INSTANCE_ID"

# Wait for instance
echo "Waiting for instance to start..."
for i in {1..120}; do
    STATUS=$(vastai show instance $INSTANCE_ID --raw 2>/dev/null | jq -r '.actual_status')
    if [ "$STATUS" = "running" ]; then
        echo "Instance is running!"
        break
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "  Status: $STATUS (${i}s)"
    fi
    sleep 1
done

# Get SSH info
SSH_HOST=$(vastai show instance $INSTANCE_ID --raw 2>/dev/null | jq -r '.ssh_host')
SSH_PORT=$(vastai show instance $INSTANCE_ID --raw 2>/dev/null | jq -r '.ssh_port')

echo "SSH: $SSH_HOST:$SSH_PORT"

# Create directory
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT root@$SSH_HOST \
    "mkdir -p /workspace/sgac_pytorch" 2>/dev/null || {
    echo "SSH connection failed. Try again in a few seconds."
    sleep 10
    ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "mkdir -p /workspace/sgac_pytorch"
}

# Copy files
echo "Uploading fixed training code..."
cd /home/it-services/ros2_ws/src/vla_6g_tvt/sgac_pytorch

scp -o StrictHostKeyChecking=no -P $SSH_PORT \
    environment_fixed.py \
    networks_fixed.py \
    sgac_agent_fixed.py \
    train_fixed.py \
    root@$SSH_HOST:/workspace/sgac_pytorch/

# Run training
echo "Starting training..."
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST << 'EOF'
cd /workspace/sgac_pytorch
pip install numpy --quiet 2>/dev/null

# Run training with verification
nohup python train_fixed.py \
    --episodes 2000 \
    --use-curriculum \
    --start-perturbation 10.0 \
    --output-dir /workspace/results_fixed \
    > /workspace/training_fixed.log 2>&1 &

echo "Training started! PID: $!"
sleep 2
head -30 /workspace/training_fixed.log 2>/dev/null || echo "(waiting for output...)"
EOF

echo ""
echo "=============================================="
echo "Training deployed!"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/training_fixed.log'"
echo ""
echo "Download results when done:"
echo "  scp -P $SSH_PORT -r root@$SSH_HOST:/workspace/results_fixed ./results_fixed_vastai"
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Don't forget to destroy when done: vastai destroy instance $INSTANCE_ID"
