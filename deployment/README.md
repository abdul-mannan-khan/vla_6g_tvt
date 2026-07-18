# Vast.ai Deployment for Math-Informed RL

## Quick Start

### 1. Set up Vast.ai API key

```bash
# Your API key is already saved at:
# ~/.config/vastai/vast_api_key
# or
# .secrets/vastai_api_key

# If not, set it up:
vastai set api-key YOUR_API_KEY
```

### 2. Deploy Training

```bash
# Default: 2000 episodes, eval every 100
./deploy_vastai.sh

# Custom: 5000 episodes, eval every 200
./deploy_vastai.sh 5000 200
```

### 3. Monitor Training

```bash
# List instances
vastai show instances

# View logs
vastai logs <instance_id>

# SSH into instance
vastai ssh <instance_id>
```

### 4. Download Results

```bash
vastai copy <instance_id>:/workspace/vla_6g_tvt/results/mi_rl/ ./results/
vastai copy <instance_id>:/workspace/vla_6g_tvt/models/mi_rl/ ./models/
```

### 5. Stop Instance

```bash
vastai stop instance <instance_id>
vastai destroy instance <instance_id>
```

## Manual Deployment

If the automated script doesn't work:

```bash
# 1. Find a GPU instance
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 dph<0.5' --limit 5

# 2. Create instance with PyTorch image
vastai create instance <offer_id> \
    --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    --disk 20

# 3. SSH in and run manually
vastai ssh <instance_id>
pip install scipy numpy matplotlib tqdm
git clone https://github.com/abdul-mannan-khan/vla_6g_tvt.git
cd vla_6g_tvt
python scripts/mi_rl/train_mi_rl.py --episodes 2000
```

## Expected Output

After training completes, you'll get:
- `results/mi_rl/mi_rl_results_TIMESTAMP.json` - Training metrics
- `results/mi_rl/sgac_model_TIMESTAMP.pt` - Trained model checkpoint

## Cost Estimate

- RTX 3090: ~$0.30-0.50/hr
- RTX 4090: ~$0.50-1.00/hr
- Training time: ~1-2 hours for 2000 episodes
- Estimated cost: $0.50-2.00
