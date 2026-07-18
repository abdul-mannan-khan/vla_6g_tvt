# VLA-6G Training on Vast.ai

Train TinyLlama on 2M samples for UAV relay positioning.

## Quick Start

### 1. Rent a GPU on Vast.ai

Go to [Vast.ai](https://vast.ai/console/create/) and rent an instance:

**Recommended specs:**
- GPU: RTX 3090 or RTX 4090 (24GB VRAM)
- CPU: 16+ cores (for data generation)
- RAM: 64GB+
- Disk: 100GB+ SSD
- Docker image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`

**Estimated costs:**
- RTX 3090: ~$0.30/hr
- RTX 4090: ~$0.50/hr
- Total for 2M samples: ~$15-25 (data gen: 4-6h, training: 8-12h)

### 2. Connect to Instance

```bash
ssh -p PORT root@IP_ADDRESS
```

### 3. Upload Files

```bash
# From your local machine
scp -P PORT -r vastai/* root@IP_ADDRESS:/workspace/
```

### 4. Run the Pipeline

```bash
cd /workspace

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (2M samples)
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh 2000000

# Or run steps separately:

# Step 1: Generate data (4-6 hours on 16 cores)
python generate_2m_data.py --num_samples 2000000 --output data/vla_2m.jsonl

# Step 2: Train (8-12 hours on RTX 3090)
python train_vla_2m.py --data_path data/vla_2m.jsonl --output_dir models/vla_2m
```

### 5. Download Trained Model

```bash
# From your local machine
scp -P PORT -r root@IP_ADDRESS:/workspace/models/vla_2m/merged ~/vla_model/
```

## Using Docker

### Build Image

```bash
docker build -t vla-6g-training .
```

### Run Container

```bash
docker run --gpus all -it -v $(pwd)/data:/workspace/data -v $(pwd)/models:/workspace/models vla-6g-training
```

### Push to Docker Hub (optional)

```bash
docker tag vla-6g-training YOUR_USERNAME/vla-6g-training:latest
docker push YOUR_USERNAME/vla-6g-training:latest
```

Then on Vast.ai, use your Docker image directly.

## Scaling Options

| Samples | Data Gen Time | Training Time | Est. Cost |
|---------|---------------|---------------|-----------|
| 100K    | 15 min        | 30 min        | $0.50     |
| 500K    | 1 hour        | 2 hours       | $1.50     |
| 1M      | 2 hours       | 4 hours       | $3.00     |
| 2M      | 4-6 hours     | 8-12 hours    | $15-25    |

## Files

- `generate_2m_data.py` - Parallel data generation using DE optimizer
- `train_vla_2m.py` - TinyLlama training with QLoRA
- `run_full_pipeline.sh` - Complete pipeline script
- `Dockerfile` - Docker image for reproducibility
- `requirements.txt` - Python dependencies

## Expected Results

With 2M training samples (vs. original 10K):
- Better generalization across topologies
- Improved throughput predictions
- More robust to edge cases

The larger dataset should help the model learn:
- Spatial relationships between users and relay
- Trade-offs between throughput, fairness, and coverage
- Optimal positioning strategies for different topologies
