#!/bin/bash
# Deploy MI-RL Demo to Vast.ai
# Creates a GPU instance and runs the demo video recording

set -e

echo "=========================================="
echo "MI-RL 6G Demo - Vast.ai Deployment"
echo "=========================================="

# Configuration
DURATION=${1:-60}
OUTPUT_FILE="mi_rl_demo_$(date +%Y%m%d_%H%M%S).mp4"

# Check for Vast.ai API key
if [ -z "$VAST_API_KEY" ]; then
    if [ -f ~/.config/vastai/vast_api_key ]; then
        export VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key)
    elif [ -f .secrets/vastai_api_key ]; then
        export VAST_API_KEY=$(cat .secrets/vastai_api_key)
    else
        echo "Error: VAST_API_KEY not set"
        exit 1
    fi
fi

echo "Duration: ${DURATION}s"
echo "Output: ${OUTPUT_FILE}"
echo ""

# Find a suitable GPU instance
echo "Searching for GPU instances..."
OFFER=$(vastai search offers 'gpu_ram>=8 num_gpus=1 dph<0.3 inet_down>=100' --limit 1 --raw 2>/dev/null | head -1)

if [ -z "$OFFER" ]; then
    echo "No suitable offers found, trying with relaxed constraints..."
    OFFER=$(vastai search offers 'num_gpus=1 dph<0.5' --limit 1 --raw 2>/dev/null | head -1)
fi

OFFER_ID=$(echo $OFFER | awk '{print $1}')
echo "Selected offer: $OFFER_ID"

# Create the startup script
STARTUP_SCRIPT=$(cat << 'EOFSCRIPT'
#!/bin/bash
set -e

# Install dependencies
pip install numpy scipy matplotlib opencv-python torch tqdm Pillow

# Clone repository
cd /workspace
git clone https://github.com/abdul-mannan-khan/vla_6g_tvt.git || true
cd vla_6g_tvt

# Create videos directory
mkdir -p videos

# Run demo recording
export PYTHONPATH=/workspace/vla_6g_tvt/scripts:$PYTHONPATH
export MPLBACKEND=Agg

python3 airsim_demo/mi_rl_demo_recorder.py \
    --checkpoint results/mi_rl/checkpoints/checkpoint_latest.pt \
    --duration DURATION_PLACEHOLDER \
    --output videos/OUTPUT_PLACEHOLDER

echo "Demo recording complete!"
ls -la videos/
EOFSCRIPT
)

# Replace placeholders
STARTUP_SCRIPT=$(echo "$STARTUP_SCRIPT" | sed "s/DURATION_PLACEHOLDER/$DURATION/g")
STARTUP_SCRIPT=$(echo "$STARTUP_SCRIPT" | sed "s/OUTPUT_PLACEHOLDER/$OUTPUT_FILE/g")

# Create instance
echo ""
echo "Creating Vast.ai instance..."
INSTANCE_ID=$(vastai create instance $OFFER_ID \
    --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    --disk 20 \
    --onstart-cmd "$STARTUP_SCRIPT" \
    2>&1 | grep -oP 'new instance is \K[0-9]+' || echo "")

if [ -z "$INSTANCE_ID" ]; then
    echo "Failed to create instance"
    exit 1
fi

echo "Instance created: $INSTANCE_ID"
echo ""

# Wait for completion
echo "Waiting for demo to complete (this may take several minutes)..."
sleep 60

# Check status
for i in {1..30}; do
    echo "  Check $i/30..."

    # Check logs
    if vastai logs $INSTANCE_ID 2>/dev/null | grep -q "Demo recording complete"; then
        echo ""
        echo "✅ Demo complete!"
        break
    fi

    sleep 30
done

# Download results
echo ""
echo "Downloading video..."
SSH_INFO=$(vastai show instance $INSTANCE_ID 2>/dev/null | grep -oP 'ssh[^\s]+')
SSH_PORT=$(echo $SSH_INFO | grep -oP ':\K[0-9]+')
SSH_HOST=$(echo $SSH_INFO | grep -oP '@\K[^:]+')

scp -P $SSH_PORT -o StrictHostKeyChecking=no \
    root@$SSH_HOST:/workspace/vla_6g_tvt/videos/$OUTPUT_FILE \
    ./videos/

echo ""
echo "=========================================="
echo "✅ Demo video saved to: videos/$OUTPUT_FILE"
echo "=========================================="

# Cleanup
echo ""
echo "Stopping instance..."
vastai stop instance $INSTANCE_ID
vastai destroy instance $INSTANCE_ID

echo "Done!"
