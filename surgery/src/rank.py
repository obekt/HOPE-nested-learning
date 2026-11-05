#!/usr/bin/env python3
"""Print a ranked comparison of every candidate across suites."""
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
FILES = sys.argv[1:] or [os.path.join(OUT, f) for f in [
    "lab1.jsonl", "lab2.jsonl", "lab3.jsonl", "lab4.jsonl", "lab5.jsonl",
    "lab6.jsonl", "lab7.jsonl", "lab8.jsonl", "eval_results.jsonl",
    "lab_confirm.jsonl", "lab_shipped.jsonl"]]
METRICS = [("ppl_wt2_test.ppl", "ppl", False), ("cloze_A.top1", "clz@1", True),
           ("cloze_A.top5", "clz@5", True), ("cloze_A.mrr", "mrr", True),
           ("mc_arc_challenge.acc_normalized", "ARC", True),
           ("mc_arc_challenge.first_tok_acc", "ARCgt", True),
           ("mc_winogrande.acc_normalized", "Wino", True),
           ("mc_truthfulqa.acc_normalized", "TQA", True),
           ("mc_hellaswag.acc_normalized", "HS", True),
           ("copy.acc_with_cue", "copy", True),
           ("ppl_wt2_val.ppl", "pplB", False), ("cloze_B.top5", "clzB@5", True),
           ("mc_mmlu.acc_normalized", "MMLU", True),
           ("mc_mmlu.first_tok_acc", "MMLUgt", True),
           ("mc_hellaswag_B.acc_normalized", "HSB", True)]

rows, seen = [], {}
for f in FILES:
    try:
        for line in open(f):
            r = json.loads(line)
            name = r.get("name") or r.get("label")
            if name in seen:
                continue
            seen[name] = 1
            r["_src"] = f
            rows.append(r)
    except FileNotFoundError:
        pass

BASE = next((r for r in rows if r.get("name") == "base:identity"
             or "foundation" in str(r.get("label", ""))), rows[0])


def val(r, key):
    v = r.get(key)
    return v if isinstance(v, (int, float)) else None


def score(r):
    """Sum of relative wins vs base on suite A (higher = better)."""
    tot = 0.0
    for k, _n, _hib in METRICS[:10]:
        v, b = val(r, k), val(BASE, k)
        if v is None or b is None:
            continue
        if k.endswith("ppl"):
            d = (b - v) / max(b, 1e-6) * 0.5          # log-ish: halve weight of ppl
        else:
            d = (v - b) / max(b, 1e-6)
        tot += max(-1.0, min(1.0, d))
    return tot


print(f"{'name':24s}" + "".join(f"{h:>7s}" for _k, h, _ in METRICS) + f"{'score':>8s}")
for r in sorted(rows, key=score, reverse=True):
    cells = ""
    for k, _h, _ in METRICS:
        v = val(r, k)
        cells += "      -" if v is None else (f"{v:>7.0f}" if k.endswith("ppl") else f"{v:>7.3f}")
    print(f"{(r.get('name') or r.get('label'))[:24]:24s}{cells}{score(r):>8.2f}")
