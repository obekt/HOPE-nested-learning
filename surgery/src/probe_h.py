#!/usr/bin/env python3
"""Representation audit: how much of the input is linearly recoverable from h?

The closed-form head fit is optimal *given* h, so these numbers explain what the
surgery can and cannot buy. Targets: the current token's embedding, the previous
token's embedding, and the decayed context bag (what the fast memory writes).
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hope_lib as hl  # noqa: E402
import surgeries as sg  # noqa: E402
import distill_head as dh  # noqa: E402

DEV = "mps"


def hidden_all(model, wins):
    with torch.no_grad():
        he = model.embedding(wins)
        fo, mem = model.fast_memory(he)
        h = model.norm_fast(he + fo)
        for l in model.cms_layers:
            h = l(h)
        return h, he, fo, mem


def r2(X, Y, n_test):
    """Ridge fit on the first windows, report held-out R^2 on the rest."""
    Xt = torch.cat([X, torch.ones(X.shape[0], 1)], 1).float()
    ntr = Xt.shape[0] - n_test
    W = sg._ridge_fit(X[:ntr], Y[:ntr].float(), 1e-3)   # adds its own intercept column
    pred = Xt[ntr:] @ W
    ss_res = ((pred - Y[ntr:].float()) ** 2).sum()
    ss_tot = ((Y[ntr:].float() - Y[:ntr].float().mean(0)) ** 2).sum()
    cos = F.cosine_similarity(pred, Y[ntr:].float(), dim=1).mean()
    return float(1 - ss_res / ss_tot), float(cos)


def audit(name, state, tiers, n=200):
    model = hl.model_from_state(state, tiers, DEV)
    wins = hl.wikitext_windows("train", n, 512).to(DEV)
    h, he, fo, mem = hidden_all(model, wins)
    E = model.embedding.weight.float()
    rows = {}
    for tag, Y in [
        ("current token e_t", he[:, 1:].reshape(-1, 512)),
        ("previous token e_{t-1}", he[:, :-1].reshape(-1, 512)),
        ("decayed context bag", None),
        ("fast-memory read (pre proj_out)", fo[:, 1:].reshape(-1, 512)),
    ]:
        if Y is None:
            # causal running mean of the embeddings before position p
            cs = he.cumsum(1)[:, :-1]
            cnt = torch.arange(1, 512, device=DEV, dtype=he.dtype).view(1, -1, 1)
            Y = (cs / cnt).reshape(-1, 512)
        X = h[:, 1:].reshape(-1, 512)
        step = max(1, X.shape[0] // 40000)
        Xs, Ys = X[::step].cpu(), Y[::step].cpu()
        nt = Xs.shape[0] // 5
        r, c = r2(Xs, Ys, nt)
        rows[tag] = {"R2": round(r, 4), "cos": round(c, 4)}
    out = {"tag": name, **rows}
    print(out, flush=True)
    del model
    return out


if __name__ == "__main__":
    res = []
    for lab, path in [("released_foundation", "surgery/downloads/hope_foundation_best.pth"),
                      ("released_final", "surgery/downloads/hope_final_best.pth")]:
        st, ck = hl.state_dict_of(path)
        res.append(audit(lab, st, ck["cms_tiers"]))
    st, ck = hl.state_dict_of("surgery/downloads/hope_foundation_best.pth")
    # head-only surgery leaves h untouched, so it cannot beat these numbers;
    # the input-side rewrites are what change what is decodable.
    variants = [
        ("surgery_embFromHead", [{"op": "emb_from_head", "params": {}}]),
        ("surgery_embFromHead+cms8off", [{"op": "emb_from_head", "params": {}},
                                        {"op": "scale_cms_mlp",
                                         "params": {"layers": [8, 9, 10, 11, 12, 13, 14, 15],
                                                    "factor": 0.0}}]),
    ]
    for name, steps in variants:
        res.append(audit(name, sg.apply_pipeline(st, steps, None), ck["cms_tiers"]))
    json_path = "surgery/out/probe_h.json"
    import json
    json.dump(res, open(json_path, "w"), indent=1)
