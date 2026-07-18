#!/usr/bin/env python3
"""
Train TinyLlama on 2M VLA Training Samples

Optimized for Vast.ai GPU instances (RTX 3090/4090 recommended).
Uses QLoRA for memory efficiency.
"""

import os
import sys
import json
import logging
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
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
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class StreamingRelayDataset(IterableDataset):
    """Streaming dataset for large JSONL files (memory efficient)."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Count lines for length estimation
        with open(data_path, 'r') as f:
            self.num_samples = sum(1 for _ in f)
        logger.info(f"Dataset has {self.num_samples:,} samples")

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        with open(self.data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                text = f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

                encodings = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )

                yield {
                    'input_ids': encodings['input_ids'].squeeze(),
                    'attention_mask': encodings['attention_mask'].squeeze(),
                    'labels': encodings['input_ids'].squeeze()
                }


class RelayDataset(Dataset):
    """Standard dataset for smaller files that fit in memory."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        logger.info(f"Loading data from {data_path}")

        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        text = f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

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
    import argparse
    parser = argparse.ArgumentParser(description='Train VLA on large dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training JSONL')
    parser.add_argument('--output_dir', type=str, default='./models/vla_2m', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use (None=all)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    args = parser.parse_args()

    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    logger.info("=" * 60)
    logger.info("VLA-6G Training on 2M Samples")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")
    logger.info(f"LoRA rank: {args.lora_r}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires a GPU.")
        sys.exit(1)

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config - higher rank for more capacity with 2M samples
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    logger.info("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    # Load dataset
    logger.info("Loading dataset...")
    dataset = RelayDataset(
        args.data_path,
        tokenizer,
        args.max_length,
        max_samples=args.max_samples
    )

    # Split into train/val
    val_size = min(5000, int(0.01 * len(dataset)))  # 1% or max 5000
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train: {train_size:,}, Validation: {val_size:,}")

    # Training arguments optimized for large dataset
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=100,
        save_steps=5000,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=2500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
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
    tokenizer.save_pretrained(args.output_dir)

    # Merge LoRA weights for faster inference
    logger.info("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_output = os.path.join(args.output_dir, "merged")
    merged_model.save_pretrained(merged_output)
    tokenizer.save_pretrained(merged_output)

    # Save training info
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "data_path": args.data_path,
            "num_samples": len(dataset),
            "num_epochs": args.epochs,
            "training_duration": str(training_duration),
            "lora_r": args.lora_r,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "completed_at": datetime.now().isoformat()
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Merged model: {merged_output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
