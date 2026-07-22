#!/bin/bash
# Automated baseline training deployment

INSTANCE_ID=45483052
MAX_WAIT=300  # 5 minutes max wait

echo "Waiting for instance $INSTANCE_ID to be ready..."
for i in $(seq 1 $MAX_WAIT); do
    STATUS=$(vastai show instance $INSTANCE_ID --raw 2>/dev/null | jq -r '.actual_status')
    if [ "$STATUS" = "running" ]; then
        echo "Instance is running!"
        break
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Status: $STATUS (waiting ${i}s)"
    fi
    sleep 1
done

# Get SSH info
SSH_HOST=$(vastai show instance $INSTANCE_ID --raw 2>/dev/null | jq -r '.ssh_host')
SSH_PORT=$(vastai show instance $INSTANCE_ID --raw 2>/dev/null | jq -r '.ssh_port')

echo "Connecting to $SSH_HOST:$SSH_PORT"

# Setup and run training
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST << 'EOF'
mkdir -p /workspace/sgac_pytorch/baselines
EOF

# Copy files
cd /home/it-services/ros2_ws/src/vla_6g_tvt/sgac_pytorch
scp -o StrictHostKeyChecking=no -P $SSH_PORT \
    environment.py networks.py \
    root@$SSH_HOST:/workspace/sgac_pytorch/

scp -o StrictHostKeyChecking=no -P $SSH_PORT \
    baselines/*.py \
    root@$SSH_HOST:/workspace/sgac_pytorch/baselines/

# Run training
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST << 'EOF'
cd /workspace/sgac_pytorch/baselines
pip install numpy --quiet
nohup python train_baselines.py --agent all --episodes 2000 --output_dir /workspace/results_baselines > /workspace/training.log 2>&1 &
echo "Training started in background. PID: $!"
EOF

echo ""
echo "Training deployed! Monitor with:"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/training.log'"
