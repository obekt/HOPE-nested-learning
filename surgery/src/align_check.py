#!/usr/bin/env python3
"""Go/no-go for embedding imputation via GPT-2.

Question: is there enough *shared* linear structure between GPT-2's token
embeddings (same 50,257 vocab, same ids) and HOPE's trained ones that a
closed-form ridge map A (768 -> 512) can predict a HOPE row from its GPT-2 row?

Reported: held-out cosine similarity of the prediction, vs two baselines —
predicting the global mean vector, and predicting a random vector.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hope_lib as hl  # noqa: E402

def gpt2_wte():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("gpt2", "pytorch_model.bin")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    return sd["wte.weight"]            # OpenAI-format key; tied input/output embedding


def ridge_map(X, Y, lam):
    """X: (n, d_in) augmented with ones; Y: (n, d_out). Returns (d_in+1, d_out)."""
    Xt = X.transpose(0, 1)
    A = Xt @ X
    A.diagonal().add_(lam * X.shape[0] / 1000.0)
    return torch.linalg.solve(A, Xt @ Y)


def evaluate(src_mat, tgt_mat, sel, name, lam=1.0):
    """Fit tgt ~ src on rows `sel`; report held-out cos on the other half."""
    torch.manual_seed(0)
    idx = sel[torch.randperm(len(sel))]
    half = len(idx) // 2
    fit, test = idx[:half], idx[half:]
    S = torch.cat([src_mat, torch.ones(src_mat.shape[0], 1)], 1).float()
    T = tgt_mat.float()
    W = ridge_map(S[fit], T[fit], lam)
    pred = S[test] @ W
    mean_pred = T[fit].mean(0, keepdim=True).expand_as(pred)
    rnd = torch.randn_like(pred)
    cos = torch.nn.functional.cosine_similarity(pred, T[test], dim=1)
    cos_m = torch.nn.functional.cosine_similarity(mean_pred, T[test], dim=1)
    cos_r = torch.nn.functional.cosine_similarity(rnd, T[test], dim=1)
    # variance explained (per-dim R^2 over the centered target)
    ss_res = ((pred - T[test]) ** 2).sum()
    ss_tot = ((T[test] - T[fit].mean(0)) ** 2).sum()
    print(f"{name:34s} n={len(idx):5d}  cos(pred)={cos.mean():+.4f}  "
          f"cos(mean-vec)={cos_m.mean():+.4f}  cos(random)={cos_r.mean():+.4f}  "
          f"R2={1 - ss_res / ss_tot:+.4f}")
    return float(cos.mean())


def main():
    ms, _ = hl.state_dict_of(os.path.join(hl.REPO, "surgery/downloads/hope_foundation_best.pth"))
    E, H, b = ms["embedding.weight"].float(), ms["head.weight"].float(), ms["head.bias"].float()
    G = gpt2_wte().float()
    print(f"GPT-2 wte {tuple(G.shape)}   HOPE E {tuple(E.shape)}  H {tuple(H.shape)}")

    ne = E.norm(dim=1)
    init_e = 0.02 * 512 ** 0.5
    # "trained" rows: moved well past the init-cluster, i.e. seen often in training
    strong = torch.nonzero(ne > 1.25 * init_e).flatten()
    mid = torch.nonzero((ne > 1.08 * init_e) & (ne <= 1.25 * init_e)).flatten()
    print(f"rows >1.25x init norm: {len(strong)}   1.08-1.25x: {len(mid)}   "
          f"<=1.08x: {50257 - len(strong) - len(mid)}")

    for lam in (0.3, 3.0, 30.0):
        print(f"\n--- ridge lambda {lam} ---")
        evaluate(G, E, strong, "G2wte -> HOPE emb (strong rows)", lam)
        evaluate(G, E, mid, "G2wte -> HOPE emb (mid rows)", lam)
        evaluate(G, H, strong, "G2wte -> HOPE head (strong rows)", lam)
        evaluate(G, H, torch.arange(0, 50257, 7), "G2wte -> HOPE head (all rows)", lam)
        evaluate(G, E, torch.arange(0, 50257, 11), "G2wte -> HOPE emb (all rows)", lam)


if __name__ == "__main__":
    main()
