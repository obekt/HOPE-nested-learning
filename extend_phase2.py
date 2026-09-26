#!/usr/bin/env python3
"""Phase-2 driver — EXPERIMENT_LOG.md decision D4 (48k marathon).

Runs after extend_training.py: re-seeds hope_final.pth from the NEW
foundation best (step=0, fresh optimizer/schedulers — same logic as
run_pipeline.py lines 64-87), then fine-tunes on the Q&A dataset for
6000 steps (2x the original 3000 budget).

Note on Phase-2 val metric: the Q&A "micro" dataset is small; with the
val stream offset most fine-tune validations may be near-empty, making
val_loss degenerate (0.0 -> best==latest). The authoritative Phase-2
evaluation is the probe suite (probe_model.py Q&A samples), per D4.
"""
import os

import torch

import train_hope as th

BEST = "hope_foundation_best.pth"
FINAL = "hope_final.pth"

if not os.path.exists(BEST):
    raise SystemExit(f"no foundation best checkpoint at {BEST}; aborting Phase 2")

ckpt = torch.load(BEST, map_location="cpu", weights_only=False)
phase2_ckpt = {
    "model_state": ckpt["model_state"],
    "step": 0,
    "best_val_loss": float("inf"),
    # optimizer/scheduler states intentionally omitted: fresh LR schedule
}
torch.save(phase2_ckpt, FINAL)
print(f"Phase 2 seeded from {BEST} (foundation step {ckpt.get('step')}, "
      f"val {ckpt.get('best_val_loss')}) -> {FINAL}", flush=True)

th.CONFIG.update({
    "learning_rate": 5e-5,
    "max_steps": 6000,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "dataset_name": "obekt/obekt-question-answer-reasoning-micro-v0.1",
    "dataset_config": None,
    "dataset_columns": "question, answer, reasoning",
    "max_samples": 80000,
    "isolate_samples": True,
    "train_skip_samples": 0,
    "val_skip_samples": 2000,   # small held-out QA slice if the dataset is big enough
    "save_path": FINAL,
    "checkpoint_every": 500,
    "log_file": "finetune_ext.log",
})

th.train()
print("PHASE 2 COMPLETE (6k) — final model: hope_final.pth")
