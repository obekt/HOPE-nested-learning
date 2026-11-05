#!/usr/bin/env python3
"""Build the report's markdown tables straight from the result files."""
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
ROWS = []
for f in ["eval_results.jsonl", "lab1.jsonl", "lab2.jsonl", "lab3.jsonl",
          "lab4.jsonl", "lab5.jsonl", "lab6.jsonl", "lab7.jsonl", "lab8.jsonl",
          "lab_confirm.jsonl", "lab_shipped.jsonl"]:
    p = os.path.join(OUT, f)
    if not os.path.exists(p):
        continue
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        r["_src"] = f
        ROWS.append(r)

IDX = {}
for r in ROWS:
    name = r.get("name") or r.get("label")
    if name and name not in IDX:
        IDX[name] = r


def g(row, key, fmt="{:.3f}", miss="–"):
    v = row.get(key)
    if v is None:
        parts = key.split(".")
        cur = row
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                cur = None
                break
        v = cur if isinstance(cur, (int, float)) else None
    if v is None:
        return miss
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


COLS = [("ppl_wt2_test.ppl", "ppl↓", "{:.0f}"),
        ("ppl_wt2_val.ppl", "pplB↓", "{:.0f}"),
        ("cloze_A.top1", "cloze@1", "{:.3f}"),
        ("cloze_A.top5", "cloze@5", "{:.3f}"),
        ("cloze_A.mrr", "clozeMRR", "{:.3f}"),
        ("cloze_B.top5", "clozeB@5", "{:.3f}"),
        ("mc_arc_challenge.acc_normalized", "ARC", "{:.3f}"),
        ("mc_winogrande.acc_normalized", "Wino", "{:.3f}"),
        ("mc_truthfulqa.acc_normalized", "TQA", "{:.3f}"),
        ("mc_hellaswag.acc_normalized", "HS", "{:.3f}"),
        ("mc_hellaswag_B.acc_normalized", "HSB", "{:.3f}"),
        ("mc_mmlu.acc_normalized", "MMLU", "{:.3f}"),
        ("copy.acc_with_cue", "copy", "{:.3f}"),
        ("copy.gain", "copyGain", "{:.3f}")]


def table(names, title):
    print(f"\n### {title}\n")
    print("| model | " + " | ".join(h for _, h, _ in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for n in names:
        r = IDX.get(n)
        if not r:
            continue
        print(f"| {n} | " + " | ".join(g(r, k, f) for k, _, f in COLS) + " |")


if __name__ == "__main__":
    groups = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
    if groups:
        for title, names in groups.items():
            table(names, title)
    else:
        table(list(IDX), "all")
