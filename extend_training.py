#!/usr/bin/env python3
"""Phase-1 extension driver — EXPERIMENT_LOG.md decision D4 (48k marathon).

Resumes foundation training from the step-12k checkpoint to max_steps=48000
with documented LR / data-protocol changes:
  - base_lrs rewritten 2e-4 -> 1e-4 in the resumed scheduler states so the
    cosine re-warm against the new horizon peaks at ~8.8e-5 (not ~1.8e-4).
  - best_val_loss reset to inf: the val split moves (skip 100k -> 600k
    articles) because extended coverage (~213k articles) would otherwise
    reach the old val region. Series are not comparable across the change;
    the extension's first validation acts as the bridge datapoint.
  - max_samples 500k -> 1.5M windows and train_skip_samples=65000 so the
    extension mostly covers NEW articles (~65k-213k) instead of re-reading
    epoch 1 a third time.
The 12k-era best checkpoint is snapshotted before anything is overwritten.
"""
import os
import shutil

import torch

import train_hope as th

MAIN = "hope_foundation.pth"
BEST = "hope_foundation_best.pth"
SNAP = "hope_foundation_best_step12k.pth"

# 1) Preserve the 12k-era best (old val protocol) before it can be overwritten
if os.path.exists(BEST) and not os.path.exists(SNAP):
    shutil.copy(BEST, SNAP)
    print(f"snapshot saved: {SNAP}", flush=True)

# 2) Patch the resume checkpoint: reset best_val_loss (val split is moving)
#    and rewrite scheduler base_lrs 2e-4 -> 1e-4 (re-warm shock mitigation).
ckpt = torch.load(MAIN, map_location="cpu", weights_only=False)
print(f"resuming from step {ckpt.get('step')} | old-protocol best val "
      f"{ckpt.get('best_val_loss'):.4f}", flush=True)
ckpt["best_val_loss"] = float("inf")
NEW_BASE_LR = 1e-4
for sch_state in ckpt.get("scheduler_states", []):
    if "base_lrs" in sch_state:
        sch_state["base_lrs"] = [NEW_BASE_LR] * len(sch_state["base_lrs"])
for opt_state in ckpt.get("optimizer_states", []):
    for pg in opt_state.get("param_groups", []):
        pg["initial_lr"] = NEW_BASE_LR
torch.save(ckpt, MAIN)
print(f"checkpoint patched: best_val_loss=inf, base_lrs={NEW_BASE_LR}", flush=True)

# 3) Extension config (D4). save_path unchanged -> train() auto-resumes.
th.CONFIG.update({
    "max_steps": 48000,
    "learning_rate": NEW_BASE_LR,
    "max_samples": 1_500_000,        # ~768M tokens ~ 213k articles (single pass)
    "train_skip_samples": 65_000,    # start near the old coverage boundary (~70k articles)
    "val_skip_samples": 600_000,     # new val split, far beyond training coverage
    "save_path": MAIN,
    "log_file": "foundation_ext.log",
    "checkpoint_every": 1000,
})

th.train()
print("PHASE 1 EXTENSION COMPLETE (48k)")
