#!/usr/bin/env python3
"""Write a surgically-edited checkpoint in the repo's own schema.

    python3 build_artifact.py --base hope_foundation_best.pth --steps '[{...}]' \
                              --out ../out/hope_x.pth --name tag --numbers eval.json

Optimizer state is dropped (the file becomes inference-only; `train_hope.py`
still resumes from it with fresh optimizers, per its own loader logic).
"""
import argparse
import json
import os
import sys
import time

import hashlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hope_lib as hl  # noqa: E402
import surgeries as sg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--other", default=None)
ap.add_argument("--steps", required=True, help="JSON list of {op,params}")
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--numbers", default=None, help="JSON file of eval metrics to embed")
ap.add_argument("--qa-format", default=None, help="true|false; default = inherit/derive")
a = ap.parse_args()


def _sha(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


ck = torch.load(a.base, map_location="cpu", weights_only=True)
base_state = {k: v.clone() for k, v in ck["model_state"].items()}
other = None
if a.other:
    ock = torch.load(a.other, map_location="cpu", weights_only=True)
    other = {k: v.clone() for k, v in ock["model_state"].items()}

steps = json.loads(a.steps) if a.steps.strip().startswith("[") else json.load(open(a.steps))
state = sg.apply_pipeline(base_state, steps, other)
n_changed = sum(1 for k in state if not (state[k] == base_state[k]).all())
l2 = sum(float((state[k].float() - base_state[k].float()).norm()) for k in state)

qa = a.qa_format
if qa is None:
    qa = bool(ck.get("qa_format", "final" in os.path.basename(a.base).lower()))

out = {
    "model_state": {k: v.contiguous() for k, v in state.items()},
    "step": ck.get("step", 0),
    "best_val_loss": ck.get("best_val_loss"),
    "cms_tiers": ck.get("cms_tiers"),
    "qa_format": bool(qa),
    "surgery": {
        "name": a.name,
        "base_checkpoint": os.path.basename(a.base),
        "base_sha256": _sha(a.base),
        "steps": steps,
        "tensors_modified": n_changed,
        "total_l2_from_base": round(l2, 4),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Weights only. No gradient step, no optimizer, no epoch was run: "
                 "every value here is either a copy, a rescale, a task-vector "
                 "combination of the two release checkpoints, or the solution of one "
                 "ridge-regression normal equation (513x513 solve)."),
    },
}
if a.numbers:
    out["surgery"]["metrics"] = json.load(open(a.numbers))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(out, a.out)
sz = os.path.getsize(a.out) / 1e6
print(f"wrote {a.out} ({sz:.0f} MB), {n_changed}/{len(state)} tensors changed, "
      f"L2 from base {l2:.1f}, qa_format={qa}")
