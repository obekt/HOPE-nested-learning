#!/usr/bin/env python3
"""How trained is each vocab row? Norm histogram + relation to the learned prior."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hope_lib as hl  # noqa: E402

ms, ck = hl.state_dict_of(os.path.join(hl.REPO, "surgery/downloads/hope_final_best.pth"))
E, H, b = ms["embedding.weight"], ms["head.weight"], ms["head.bias"]
ne, nh = E.norm(dim=1), H.norm(dim=1)
init_e, init_h = 0.02 * 512 ** 0.5, (6 / (50257 + 512)) ** 0.5 / 3 ** 0.5 * 512 ** 0.5
print(f"init norms: embedding {init_e:.4f}  head {init_h:.4f}")

for name, n, init in [("embedding", ne, init_e), ("head", nh, init_h)]:
    q = torch.quantile(n, torch.linspace(0, 1, 13))
    print(f"\n{name} row-norm deciles (init={init:.4f}):")
    print("  " + " ".join(f"{v:.3f}" for v in q))
    ratio = n / init
    hist = torch.histc(ratio.float(), bins=20, min=0.0, max=4.0)
    for i, h in enumerate(hist):
        lo = i * 0.2
        print(f"   {lo:.1f}-{lo+0.2:.1f}: {int(h):6d} {'#' * min(60, int(h / max(hist.max(),1) * 60))}")

# relation to the learned prior (proxy for training-corpus frequency)
r_e = np.corrcoef(ne.numpy(), b.numpy())[0, 1]
r_h = np.corrcoef(nh.numpy(), b.numpy())[0, 1]
print(f"\ncorr(||E_row||, head.bias) = {r_e:.3f}\ncorr(||H_row||, head.bias) = {r_h:.3f}")
# cos similarity between embedding and head rows (weight tying signal)
cs = torch.nn.functional.cosine_similarity(
    E / E.norm(dim=1, keepdim=True), H / H.norm(dim=1, keepdim=True), dim=1)
print(f"cos(E_row, H_row): mean {cs.mean():.3f} median {cs.median():.3f} "
      f"p10 {cs.quantile(0.1):.3f} p90 {cs.quantile(0.9):.3f}")
# frequent vs rare split by prior
hi = b > b.quantile(0.9)
lo = b < b.quantile(0.5)
print(f"top-decile-prior rows: mean ||E||={ne[hi].mean():.4f} ||H||={nh[hi].mean():.4f}")
print(f"bottom-half-prior rows: mean ||E||={ne[lo].mean():.4f} ||H||={nh[lo].mean():.4f}")

# is the embedding still ~iid gaussian for low-prior rows?
sub = E[lo]
print(f"low-prior embedding rows: std {sub.std():.5f} (init 0.0200), "
      f"mean {sub.mean():.6f}, kurtosis-ish {(sub**4).mean()/(sub**2).mean()**2:.3f} (gaussian=3.0)")
subh = H[lo]
print(f"low-prior head rows:      std {subh.std():.5f} (xavier-init {init_h/512**0.5:.5f}), "
      f"gaussian check {(subh**4).mean()/(subh**2).mean()**2:.3f}")
