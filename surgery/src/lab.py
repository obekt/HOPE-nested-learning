#!/usr/bin/env python3
"""Screen weight-surgery candidates against the pre-registered suite.

    python3 lab.py --base <pth> --other <pth> --specs specs.json --suites A --out run.jsonl

specs.json: [{"name": "...", "steps": [{"op": "...", "params": {...}}]}, ...]
Results are appended as JSON lines; a compact table is printed as they land.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as es  # noqa: E402
import hope_lib as hl  # noqa: E402
import surgeries as sg  # noqa: E402

KEY = [("ppl_wt2_test", "ppl"), ("cloze_A", "top1"), ("cloze_A", "top5"),
       ("cloze_A", "mrr"), ("mc_arc_challenge", "acc_normalized"),
       ("mc_arc_challenge", "first_tok_acc"), ("mc_winogrande", "acc_normalized"),
       ("mc_truthfulqa", "acc_normalized"), ("mc_hellaswag", "acc_normalized"),
       ("copy", "acc_with_cue")]
KEYB = [("ppl_wt2_val", "ppl"), ("cloze_B", "top1"), ("cloze_B", "top5"),
        ("mc_mmlu", "acc_normalized"), ("mc_mmlu", "first_tok_acc"),
        ("mc_hellaswag_B", "acc_normalized")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", default=None)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--suites", default="A")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base, bck = hl.state_dict_of(a.base)
    other = hl.state_dict_of(a.other)[0] if a.other else None
    specs = json.load(open(a.specs))
    tiers = bck.get("cms_tiers")
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass
    cols = KEY + (KEYB if "B" in a.suites else [])
    heads = ["ppl", "clz@1", "clz@5", "mrr", "ARC", "ARCgt", "Wino", "TQA", "HS", "copy"]
    headsB = ["pplB", "clzB1", "clzB5", "MMLU", "MMLUgt", "HSB"]
    hh = heads + (headsB if "B" in a.suites else [])
    print("name" + "".join(f"{h:>10s}" for h in hh), flush=True)
    for sp in specs:
        if sp["name"] in done:
            print(f"[skip] {sp['name']}", flush=True)
            continue
        t0 = time.time()
        st = sg.apply_pipeline(base, sp["steps"], other)
        d = sum(float((st[k].float() - base[k].float()).norm()) for k in st)
        try:
            r = es.evaluate_state(st, a.device, a.suites,
                                  meta={"label": sp["name"]}, tiers=tiers)
        except Exception as e:
            print(f"[FAIL] {sp['name']}: {type(e).__name__}: {e}", flush=True)
            continue
        rec = {"name": sp["name"], "steps": sp["steps"], "l2_from_base": round(d, 3),
               "secs": round(time.time() - t0, 1)}
        for m, k in cols:
            if m in r and isinstance(r[m], dict) and k in r[m]:
                rec[f"{m}.{k}"] = r[m][k]
        rec["qa"] = r.get("qa_A", {}).get("samples", r.get("qa_B", {}).get("samples", []))
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        row = sp["name"][:26] + "".join(
            f"{rec.get(m + '.' + k, float('nan')):>10.3f}" for m, k in cols)
        print(row[:240], f"  [{rec['secs']}s]", flush=True)


if __name__ == "__main__":
    main()
