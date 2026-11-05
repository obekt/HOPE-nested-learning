#!/usr/bin/env python3
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as es  # noqa: E402
import hope_lib as hl  # noqa: E402

print("tok check:", hl.tok("Hello world") == hl.tok_ids("Hello world"),
      "| call-vs-encode:", hl.th.TOKENIZER("Hello world")["input_ids"] == hl.tok("Hello world"))

path = sys.argv[1]
dev = sys.argv[2] if len(sys.argv) > 2 else "mps"
t0 = time.time()
model, ck = hl.load_model(path, device=dev)
print(f"loaded {ck.get('step')} in {time.time()-t0:.1f}s on {dev}; "
      f"head.weight dtype {dict(model.named_parameters())['head.weight'].dtype}")

t = time.time(); print("ppl:", es.lm_ppl(model, dev, n_windows=4), f"{time.time()-t:.1f}s")
t = time.time()
m = es.cloze_metrics(model, dev, es.SPEC["cloze_repo_A"][:5])
print("cloze5:", {k: v for k, v in m.items() if k != "items"}, f"{time.time()-t:.1f}s")
t = time.time()
items = es.build_mc_items("arc_challenge", {"split": "test", "limit": 10})
print("mc arc10:", es.mc_suite(model, dev, items, "arc"), f"{time.time()-t:.1f}s")
t = time.time()
wino = es.build_mc_items("winogrande", {"name": "allenai/winogrande", "config": "winogrande_xl",
                                        "split": "validation", "limit": 10})
print("mc wino10:", es.mc_suite(model, dev, wino, "wino"), f"{time.time()-t:.1f}s")
t = time.time()
tq = es.build_mc_items("truthfulqa", {"split": "mc1", "limit": 10})
print("mc tqa10:", es.mc_suite(model, dev, tq, "tqa"), f"{time.time()-t:.1f}s")
t = time.time()
print("copy20:", es.copy_gain(model, dev, 20), f"{time.time()-t:.1f}s")
t = time.time()
print("qa:", json.dumps(es.qa_greedy(model, dev, ["Question: Why is the sky blue?\nAnswer:"])))
print(f"{time.time()-t:.1f}s")
t = time.time()
hs = es.build_mc_items("hellaswag", {"name": "Rowan/hellaswag", "split": "validation", "limit": 10, "seed": 0})
print("mc hs10:", es.mc_suite(model, dev, hs, "hs"), f"{time.time()-t:.1f}s (incl dataset load)")
t = time.time()
mm = es.build_mc_items("mmlu", {"split": "test", "subjects": ["astronomy"], "per_subject": 10})
print("mc mmlu10:", es.mc_suite(model, dev, mm, "mmlu"), f"{time.time()-t:.1f}s")
