# MI-RL Paper - Continuation Notes

## Last Session Summary (2026-07-21)

### Completed Tasks:

1. **Paper Improvements** (Committed: 97f246a):
   - Expanded introduction from 3 to 17 paragraphs
   - Added comprehensive symbol definitions
   - Updated references from 8 to 30 citations

2. **MATLAB Demo Script** (Committed: 6e4ff5b):
   - Created simplified MATLAB visualization (not real training)
   - Results showed ~21% improvement vs random

3. **PyTorch SGAC Implementation** (Committed: e59d06d):
   - `sgac_pytorch/environment.py` - UAV relay environment with SCA solver
   - `sgac_pytorch/networks.py` - Actor, Twin Critic, SGAC policy, Replay buffer
   - `sgac_pytorch/sgac_agent.py` - Full TD3-style SGAC agent
   - `sgac_pytorch/train.py` - Training script with evaluation
   - `sgac_pytorch/deploy_vastai.sh` - Deployment script

### Active Training on Vast.ai:

**Instance:** 45466580
- **GPU:** RTX 4090
- **Location:** Netherlands
- **Price:** $0.35/hr
- **Status:** Running

**Monitor with:**
```bash
export VAST_API_KEY=$(cat .secrets/vastai_api_key)
vastai logs 45466580 --tail 100
vastai show instances
```

**Download results when complete:**
```bash
vastai copy 45466580:/workspace/results ./results_vastai/
```

**Stop/destroy instance:**
```bash
vastai stop instance 45466580
vastai destroy instance 45466580
```

### Training Configuration:
- Episodes: 2000
- Scenarios: 100 training, 50 evaluation
- Batch size: 256
- Learning rate: 3e-4
- SGAC weights: alpha=0.7, beta=0.3

### Early Baseline Results:
- Random: 5050 Mbps
- Analytical: 5532 Mbps
- SCA: 5866 Mbps

### Key Files:
- Paper: `/home/it-services/ros2_ws/src/vla_6g_tvt/paper/mirl_iot_journal.tex`
- PyTorch: `/home/it-services/ros2_ws/src/vla_6g_tvt/sgac_pytorch/`
- MATLAB: `/home/it-services/ros2_ws/src/vla_6g_tvt/matlab/sgac_training.m`

### Git Commits (This Session):
1. `97f246a` - Expand introduction, add symbol definitions, update references
2. `6e4ff5b` - Add MATLAB SGAC training and simulation script
3. `44a2610` - Update continuation notes with MATLAB simulation results
4. `e59d06d` - Add comprehensive PyTorch SGAC implementation

### Next Steps:
1. Monitor Vast.ai training progress
2. Download results when training completes
3. Generate proper figures from training data
4. Update paper with real experimental results
