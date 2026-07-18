#!/bin/bash
# Full VLA Training Pipeline for Vast.ai
#
# Usage: ./run_full_pipeline.sh [num_samples]
# Default: 2000000 samples

set -e

NUM_SAMPLES=${1:-2000000}
DATA_FILE="data/vla_training_${NUM_SAMPLES}.jsonl"
MODEL_DIR="models/vla_${NUM_SAMPLES}"

echo "============================================================"
echo "VLA-6G Full Training Pipeline"
echo "============================================================"
echo "Samples: ${NUM_SAMPLES}"
echo "Data file: ${DATA_FILE}"
echo "Model output: ${MODEL_DIR}"
echo "============================================================"

# Create directories
mkdir -p data models

# Step 1: Generate training data
echo ""
echo "[Step 1/3] Generating training data..."
echo "============================================================"

if [ -f "${DATA_FILE}" ]; then
    EXISTING_LINES=$(wc -l < "${DATA_FILE}")
    echo "Found existing data file with ${EXISTING_LINES} samples"
    if [ "${EXISTING_LINES}" -ge "${NUM_SAMPLES}" ]; then
        echo "Skipping data generation (already have enough samples)"
    else
        echo "Continuing data generation from ${EXISTING_LINES}..."
        python generate_2m_data.py \
            --num_samples ${NUM_SAMPLES} \
            --output ${DATA_FILE} \
            --start_seed ${EXISTING_LINES}
    fi
else
    python generate_2m_data.py \
        --num_samples ${NUM_SAMPLES} \
        --output ${DATA_FILE}
fi

# Verify data
ACTUAL_LINES=$(wc -l < "${DATA_FILE}")
echo "Data file has ${ACTUAL_LINES} samples"

# Step 2: Train TinyLlama
echo ""
echo "[Step 2/3] Training TinyLlama with QLoRA..."
echo "============================================================"

python train_vla_2m.py \
    --data_path ${DATA_FILE} \
    --output_dir ${MODEL_DIR} \
    --epochs 1 \
    --batch_size 8 \
    --grad_accum 4 \
    --lora_r 64 \
    --lr 2e-4

# Step 3: Evaluate
echo ""
echo "[Step 3/3] Evaluation..."
echo "============================================================"

# Quick evaluation on a few samples
python -c "
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '${MODEL_DIR}/merged'
print(f'Loading model from {model_path}...')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto'
)

# Test inference
test_prompt = '''### Instruction:
You are a 6G UAV relay positioning expert. Given the current network state, determine the optimal 3D position for the UAV relay to maximize throughput and fairness.

Current State:
- Base station: (0.0, 0.0, 30.0)
- UAV position: (25.0, 25.0, 20.0)
- Total throughput: 80.0 Mbps
- Fairness index: 0.900

Ground Users (5 total):
  User 0: position=(50.0, 50.0), SNR=-25.0dB, rate=20.0Mbps
  User 1: position=(60.0, 40.0), SNR=-26.0dB, rate=18.0Mbps
  User 2: position=(40.0, 60.0), SNR=-27.0dB, rate=16.0Mbps
  User 3: position=(55.0, 55.0), SNR=-26.5dB, rate=17.0Mbps
  User 4: position=(45.0, 45.0), SNR=-24.0dB, rate=22.0Mbps

Provide the optimal UAV position and explain your reasoning.

### Response:
'''

inputs = tokenizer(test_prompt, return_tensors='pt').to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.1,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print('Test inference result:')
print(response.split('### Response:')[1].strip()[:200])
print('...')
"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo "Model saved to: ${MODEL_DIR}"
echo "Merged model: ${MODEL_DIR}/merged"
echo ""
echo "To download the model:"
echo "  scp -r ${MODEL_DIR}/merged your_machine:~/vla_model/"
echo "============================================================"
