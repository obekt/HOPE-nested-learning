#!/usr/bin/env python3
"""Weight-space + function-space diagnostics. Read-only.

Answers: which parts of this binary are actually trained, which are still at
init, how much each CMS layer contributes, and how much of the output is just
the learned unigram prior.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hope_lib as hl  # noqa: E402
import eval_suite as es  # noqa: E402

DEV = "mps"

def init_row_norm(kind, out_f, in_f):
    """Expected L2 norm of a nn.Linear row at xavier init (row = in_f elements)."""
    bound = math.sqrt(6.0 / (out_f + in_f))
    return bound / math.sqrt(3.0) * math.sqrt(in_f)


def col_norms_for_embedding():
    return 0.02 * math.sqrt(512)      # nn.Embedding normal(0, 0.02)


def analyze(path, tag):
    ms, ck = hl.state_dict_of(path)
    out = {"tag": tag, "ckpt": os.path.basename(path), "step": ck.get("step")}

    # ---- 1. how many vocab rows are still at init?
    for name, expect_norm in [("embedding.weight", col_norms_for_embedding()),
                              ("head.weight", init_row_norm("linear", 50257, 512))]:
        W = ms[name]
        nrm = W.norm(dim=1)
        d = (nrm - expect_norm).abs() / expect_norm
        out[name] = {
            "init_norm": round(expect_norm, 4),
            "norm_mean": round(float(nrm.mean()), 4),
            "norm_median": round(float(nrm.median()), 4),
            "norm_p05": round(float(nrm.quantile(0.05)), 4),
            "norm_p95": round(float(nrm.quantile(0.95)), 4),
            "frac_within_10pct_of_init": round(float((d < 0.10).float().mean()), 4),
            "frac_within_25pct_of_init": round(float((d < 0.25).float().mean()), 4),
            "frac_shrunk_below_60pct_init": round(float((nrm < 0.6 * expect_norm).float().mean()), 4),
            "frac_grown_above_2x_init": round(float((nrm > 2 * expect_norm).float().mean()), 4),
        }

    # ---- 2. per-tensor drift vs init scale (rms / init_rms)
    drift = {}
    for k, v in ms.items():
        if v.ndim < 2:
            continue
        o, i = v.shape[0], v.shape[1] if v.ndim >= 2 else v.shape[0]
        exp = init_row_norm("linear", o, i) / math.sqrt(i) if "embedding" not in k else 0.02
        rms = float(v.float().pow(2).mean().sqrt())
        drift[k] = round(rms / exp, 3)
    out["rms_over_init"] = {k: drift[k] for k in sorted(drift) if drift[k] < 1.6 or drift[k] > 3}
    out["cms_layer_drift"] = [drift.get(f"cms_layers.{l}.net.0.weight") for l in range(16)]

    # ---- 3. head bias = learned prior?
    hb = ms["head.bias"]
    out["head_bias"] = {"min": round(float(hb.min()), 3), "max": round(float(hb.max()), 3),
                        "mean": round(float(hb.mean()), 3), "std": round(float(hb.std()), 3),
                        "top20": [hl.th.TOKENIZER.decode([i]).strip()
                                  for i in torch.topk(hb, 20).indices],
                        "bottom20": [hl.th.TOKENIZER.decode([i]).strip()
                                     for i in torch.topk(hb, 20, largest=False).indices]}

    # ---- 4. function space: layer contributions + gates + prior dominance
    model, _ = hl.load_model(path, device=DEV)
    wins = hl.wikitext_windows("test", 8, 512).to(DEV)
    with torch.no_grad():
        h = model.embedding(wins)
        fast_out, mem = model.fast_memory(h)
        contrib_fast = float(fast_out.norm() / h.norm())
        a_pre = model.fast_memory.gate_alpha(h) + 4.0
        b = torch.sigmoid(model.fast_memory.gate_beta(h) - 2.0)
        alpha = torch.sigmoid(a_pre)
        out["fast_memory"] = {
            "contrib_ratio": round(contrib_fast, 4),
            "alpha_mean": round(float(alpha.mean()), 5), "alpha_p05": round(float(alpha.quantile(0.05)), 5),
            "alpha_p95": round(float(alpha.quantile(0.95)), 5),
            "beta_mean": round(float(b.mean()), 5), "beta_p95": round(float(b.quantile(0.95)), 5),
            "mem_norm_end": round(float(mem[:, -1, :].norm(dim=1).mean()), 4),
            "mem_norm_over_sqrtD": round(float(mem.norm() / (mem.shape[0] * mem.shape[1] ** 0.5)), 4),
        }
        hh = model.norm_fast(h + fast_out)
        layer_contrib, layer_cos = [], []
        x = hh
        for l in model.cms_layers:
            y = l(x)
            layer_contrib.append(round(float((y - x).norm() / x.norm()), 4))
            layer_cos.append(round(float(F.cosine_similarity(y.flatten(), x.flatten(), dim=0)), 4))
            x = y
        out["cms_residual_contrib"] = layer_contrib
        logits = model.head(x)
        prior = ms["head.bias"].to(DEV)
        # how often does the prior alone match the model's argmax?
        am_model = logits.argmax(-1)
        am_prior = prior.argmax(-1).expand_as(am_model)
        out["prior_agreement"] = round(float((am_model == am_prior).float().mean()), 4)
        p_model = F.softmax(logits.float(), -1)
        p_prior = F.softmax(prior, -1)
        out["kl_model_vs_prior"] = round(float((p_model * (p_model.clamp_min(1e-12).log()
                                             - p_prior.log())).sum(-1).mean()), 4)
        # logit scale actually used
        out["logit_std"] = round(float(logits.float().std()), 3)
        out["ppl_mps_check"] = es.lm_ppl(model, DEV, n_windows=8)
    return out


def main():
    res = []
    for lab, path in [("final", os.path.join(hl.REPO, "surgery/downloads/hope_final_best.pth")),
                      ("foundation", os.path.join(hl.REPO, "surgery/downloads/hope_foundation_best.pth"))]:
        print(f"=== {lab} ===", flush=True)
        r = analyze(path, lab)
        res.append(r)
        print(json.dumps(r, indent=1), flush=True)
    # task-vector structure between the two releases
    ms_f, _ = hl.state_dict_of(os.path.join(hl.REPO, "surgery/downloads/hope_final_best.pth"))
    ms_b, _ = hl.state_dict_of(os.path.join(hl.REPO, "surgery/downloads/hope_foundation_best.pth"))
    tv = {}
    for k in ms_f:
        d = (ms_f[k].float() - ms_b[k].float())
        tv[k] = {"l2": round(float(d.norm()), 3),
                 "rel": round(float(d.norm() / (ms_b[k].float().norm() + 1e-9)), 4)}
    print("\n=== task vector (final - foundation), largest relative movers ===", flush=True)
    for k, v in sorted(tv.items(), key=lambda kv: -kv[1]["rel"])[:25]:
        print(f"  {k:38s} rel={v['rel']:7.4f} l2={v['l2']:9.2f}", flush=True)
    print("\n=== most FROZEN tensors (tiny relative movement) ===", flush=True)
    for k, v in sorted(tv.items(), key=lambda kv: kv[1]["rel"])[:20]:
        print(f"  {k:38s} rel={v['rel']:7.4f} l2={v['l2']:9.2f}", flush=True)
    res.append({"task_vector": tv})
    os.makedirs(hl.OUT, exist_ok=True)
    json.dump(res, open(os.path.join(hl.OUT, "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
