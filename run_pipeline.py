#!/usr/bin/env python3
"""
Full training pipeline: Foundation (Wikipedia) -> Fine-tuning (Q&A)
Runs both phases automatically.
"""
import os
import sys
import time

# ==========================================
# PHASE 1: FOUNDATION TRAINING
# ==========================================
print("=" * 60)
print("PHASE 1: FOUNDATION TRAINING (Wikipedia)")
print("=" * 60)

# Override train_hope CONFIG for Wikipedia
import train_hope as th

th.CONFIG.update({
    "d_model": 512,
    "n_layers": 16,
    "vocab_size": 256,
    "seq_len": 512,
    "batch_size": 4,
    "accumulate_grad": 8,
    "learning_rate": 2e-4,
    "max_steps": 12000,
    "warmup_steps": 1500,
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    "dataset_name": "wikimedia/wikipedia",
    "dataset_config": "20231101.en",
    "dataset_columns": "title, text",
    "max_samples": 500000,
    "isolate_samples": False,
    "save_path": "hope_foundation.pth",
    "checkpoint_every": 1000,
    "log_file": "foundation.log",
})

# Run foundation training
th.train()

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE!")
print("=" * 60)

# ==========================================
# PHASE 2: FINE-TUNING ON Q&A
# ==========================================
print("\n" + "=" * 60)
print("PHASE 2: FINE-TUNING (Q&A Micro Dataset)")
print("=" * 60)

# Reset training state for fine-tuning
th.CONFIG.update({
    "learning_rate": 5e-5,  # Much lower for fine-tuning
    "max_steps": 3000,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "dataset_name": "obekt/obekt-question-answer-reasoning-micro-v0.1",
    "dataset_config": None,
    "dataset_columns": "question, answer, reasoning",
    "max_samples": 80000,
    "isolate_samples": True,
    "save_path": "hope_final.pth",
    "checkpoint_every": 500,
    "log_file": "finetune.log",
})

# The train() function will auto-load hope_foundation.pth if it exists
# and continue training on the new dataset
th.train()

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE! Final model: hope_final.pth")
print("=" * 60)
