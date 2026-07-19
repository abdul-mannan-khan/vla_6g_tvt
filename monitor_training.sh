#!/bin/bash
# Monitor MI-RL training progress every 30 minutes

INSTANCE_ID="45273768"
LOG_FILE="/home/it-services/ros2_ws/src/vla_6g_tvt/training_monitor.log"
API_KEY=$(cat /home/it-services/ros2_ws/src/vla_6g_tvt/.secrets/vastai_api_key)

export VAST_API_KEY="$API_KEY"

echo "======================================" >> "$LOG_FILE"
echo "Training Monitor Started: $(date)" >> "$LOG_FILE"
echo "Instance: $INSTANCE_ID" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

check_progress() {
    echo "" >> "$LOG_FILE"
    echo "--- Check at $(date) ---" >> "$LOG_FILE"
    
    # Get instance status
    STATUS=$(vastai show instance $INSTANCE_ID 2>/dev/null | tail -1)
    echo "Status: $STATUS" >> "$LOG_FILE"
    
    # Get recent training logs (last 30 lines with episode info)
    echo "Recent progress:" >> "$LOG_FILE"
    vastai logs $INSTANCE_ID --tail 50 2>/dev/null | grep -E "(Episode|throughput|SGAC|Checkpoint|Training|Best)" >> "$LOG_FILE"
    
    # Check if training completed
    if vastai logs $INSTANCE_ID 2>/dev/null | grep -q "Training complete"; then
        echo "*** TRAINING COMPLETED ***" >> "$LOG_FILE"
        return 1
    fi
    
    return 0
}

# Initial check after 5 minutes
sleep 300
check_progress

# Then check every 30 minutes
while true; do
    sleep 1800  # 30 minutes
    if ! check_progress; then
        echo "Training finished. Stopping monitor." >> "$LOG_FILE"
        break
    fi
done
