#!/usr/bin/env python3
"""Structural + statistical inspection of a HOPE checkpoint, read-only."""
import sys
import torch

path = sys.argv[1]
try:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    mode = "weights_only=True"
except Exception as e:
    print(f"[weights_only failed: {type(e).__name__}: {e}]")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mode = "weights_only=False"
print(f"loaded with {mode}")
print("top-level keys:", list(ck.keys()))
for k, v in ck.items():
    if k != "model_state":
        if isinstance(v, (list, tuple)):
            print(f"  {k}: list[{len(v)}] of {type(v[0]).__name__ if v else 'empty'}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys {list(v.keys())[:8]}")
        else:
            print(f"  {k}: {v!r}")

ms = ck["model_state"]
print(f"\nmodel_state: {len(ms)} tensors, "
      f"params={sum(t.numel() for t in ms.values()):,}")
dtypes = {}
for k, t in ms.items():
    dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + t.numel()
print("dtype share (by numel):", dtypes)

print("\n{:38s} {:>16s} {:>8s} {:>9s} {:>9s} {:>9s}".format(
    "tensor", "shape", "std", "absmean", "rms", "max|.|"))
def stats(t):
    tf = t.float()
    return (tf.std().item(), tf.abs().mean().item(), tf.norm().item() / (tf.numel() ** 0.5), tf.abs().max().item())

order = list(ms.keys())
# group print
for k in order:
    t = ms[k]
    s = stats(t)
    print(f"{k:38s} {str(tuple(t.shape)):>16s} {s[0]:8.4f} {s[1]:9.5f} {s[2]:9.5f} {s[3]:9.4f}")
