#!/usr/bin/env python3
"""What is the ceiling for a closed-form linear readout of this architecture?

Identical objective to the shipped surgery (least squares onto GPT-2's
log-probabilities, so a perfect fit reproduces the teacher's distribution), but
allowed to read feature blocks the architecture does not give the head:

  h              what HOPE's head actually sees
  h + bag        + causal running mean of the context embeddings (the fast
                 memory's raw information, undistorted by the 16 position-wise
                 CMS blocks)
  h + bag + prev + the previous token's embedding

Held-out CE is always against the true next token on wikitext-2 test. The gap
between row 1 and row 2/3 is what the *architecture*, not the solve, costs.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hope_lib as hl  # noqa: E402

DEV = "mps"
PATH = sys.argv[1] if len(sys.argv) > 1 else "surgery/downloads/hope_foundation_best.pth"
N_FIT_W, STRIDE, N_TEST_W = 150, 4, 24

state, ck = hl.state_dict_of(PATH)
model = hl.model_from_state(state, ck["cms_tiers"], DEV)
from transformers import AutoModelForCausalLM  # noqa: E402
teacher = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).to(DEV).eval()


def blocks(w):
    with torch.no_grad():
        he = model.embedding(w)
        fo, _ = model.fast_memory(he)
        h = model.norm_fast(he + fo)
        for l in model.cms_layers:
            h = l(h)
        lp = F.log_softmax(teacher(w).logits[:, :-1].float(), -1).clamp_min(-25)
    cs = he.cumsum(1)[:, :-1]
    cnt = torch.arange(1, w.shape[1], device=DEV, dtype=he.dtype).view(1, -1, 1)
    prev = torch.cat([torch.zeros_like(he[:, :1]), he[:, :-2]], 1)   # e_{t-1}
    return {"h": h[:, :-1], "bag": cs / cnt, "prev": prev, "y": lp, "tgt": w[:, 1:]}


def collect(parts, w):
    b = blocks(w.to(DEV))
    X = torch.cat([b[p] for p in parts], -1)
    sel = torch.zeros(X.shape[1], dtype=torch.bool, device=DEV)
    sel[::STRIDE] = True
    m = sel.unsqueeze(0) & (b["tgt"] != hl.EOS)
    d = X.shape[-1]
    return (X[m].reshape(-1, d).cpu(), b["y"][m].reshape(-1, hl.VOCAB).cpu(),
            b["tgt"][m].reshape(-1).cpu())


def run(parts, label):
    Xs, Ys, Ts = [], [], []
    FIT = hl.wikitext_windows("train", N_FIT_W, 512)
    TEST = hl.wikitext_windows("test", N_TEST_W, 512)
    for i in range(0, N_FIT_W, 20):
        x, y, t = collect(parts, FIT[i:i + 20])
        Xs.append(x); Ys.append(y); Ts.append(t)
    X = torch.cat(Xs).double(); Y = torch.cat(Ys).double()
    Xt_ = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
    A = Xt_.T @ Xt_
    B = Xt_.T @ Y
    W = torch.linalg.solve(A + 1e-3 * A.diagonal().mean() * torch.eye(A.shape[0], dtype=A.dtype), B)
    xe, ye, te = [], [], []
    for i in range(0, N_TEST_W, 10):
        x, y, t = collect(parts, hl.wikitext_windows("test", i + 10, 512)[i:i + 10])
        xe.append(x); te.append(t)
    Xe = torch.cat(xe).to(DEV).float(); Te = torch.cat(te).to(DEV)
    logits = (torch.cat([Xe, torch.ones(Xe.shape[0], 1, device=DEV)], 1) @ W.to(DEV).float()).cpu()
    ce = F.cross_entropy(logits, Te).item()
    # also: how well does the fit track the teacher itself?
    Ye = torch.cat(ye).to(DEV).float()
    kl = (Ye.exp() * (Ye - logits.to(DEV).log_softmax(-1))).sum(-1).mean().item()
    print(f"{label:16s} dims={X.shape[1]:5d} n={X.shape[0]:7d} "
          f"heldout_CE={ce:.4f} ppl={torch.tensor(ce).exp():8.1f} KL_to_teacher={kl:.3f}",
          flush=True)
    return {"feats": label, "dims": int(X.shape[1]), "n": int(X.shape[0]),
            "ce": round(ce, 4), "ppl": round(float(torch.tensor(ce).exp()), 1),
            "kl_teacher": round(kl, 3)}


if __name__ == "__main__":
    out = [run(["h"], "h"), run(["h", "bag"], "h+bag"),
           run(["h", "bag", "prev"], "h+bag+prev"), run(["bag"], "bag only"),
           run(["prev"], "prev only")]
    import json
    json.dump(out, open("surgery/out/oracle_ce.json", "w"), indent=1)
