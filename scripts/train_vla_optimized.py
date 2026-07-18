#!/usr/bin/env python3
"""
Optimized VLA Training Script for RTX 4050 (6GB VRAM)

Uses:
- 4-bit quantization (QLoRA)
- Gradient checkpointing
- Small batch size with gradient accumulation
- Reduced sequence length
"""

import os
import sys
import json
import logging
from datetime import datetime

# Set memory optimization before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/it-services/ros2_ws/src/vla_6g_tvt/models/training.log')
    ]
)
logger = logging.getLogger(__name__)


class RelayDataset(Dataset):
    """Dataset for relay positioning training."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        logger.info(f"Loading data from {data_path}")

        with open(data_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Format as chat/instruction format
        text = f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }


def main():
    # Configuration optimized for 6GB VRAM
    # Using TinyLlama - open model, no authentication required
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DATA_PATH = "/home/it-services/ros2_ws/src/vla_6g_tvt/data/llama_finetune_20260130_202752.jsonl"
    OUTPUT_DIR = "/home/it-services/ros2_ws/src/vla_6g_tvt/models/vla_6g_v1"

    # Training hyperparameters for 6GB GPU
    MAX_LENGTH = 384
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 3
    LORA_R = 16
    LORA_ALPHA = 32

    logger.info("=" * 60)
    logger.info("VLA-6G Relay Training (Optimized for RTX 4050 6GB)")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Data: {DATA_PATH}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("CUDA not available — training on CPU (slower)")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_gpu:
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Loading model for CPU training...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    logger.info("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Load dataset with 10% validation split
    logger.info("Loading dataset...")
    full_dataset = RelayDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    # Limit dataset size for CPU training
    max_samples = 2000 if not use_gpu else len(full_dataset)
    if len(full_dataset) > max_samples:
        full_dataset, _ = torch.utils.data.random_split(
            full_dataset, [max_samples, len(full_dataset) - max_samples],
            generator=torch.Generator().manual_seed(42)
        )
        logger.info(f"Using {max_samples} samples (CPU mode)")
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train: {train_size}, Validation: {val_size}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        fp16=use_gpu,
        optim="paged_adamw_8bit" if use_gpu else "adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        use_cpu=not use_gpu,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    training_duration = end_time - start_time
    logger.info(f"Training completed in {training_duration}")

    # Save final model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training info
    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "data_path": DATA_PATH,
            "num_samples": len(train_dataset),
            "num_epochs": NUM_EPOCHS,
            "training_duration": str(training_duration),
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "max_length": MAX_LENGTH,
            "completed_at": datetime.now().isoformat()
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
