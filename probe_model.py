#!/usr/bin/env python3
"""
Probe harness: measures what HOPE has learned at each checkpoint, WITHOUT
touching the running training (loads a copy of the checkpoint on CPU).

Appends one JSON line per probe run to probe_stats.jsonl (skips checkpoints
already probed at the same step). Run manually or from cron:

    python3 probe_model.py            # probe any checkpoint with a new step
    python3 probe_model.py --force    # re-probe even if step already recorded
    python3 probe_model.py --report   # print accumulated stats table, no probing

WOW criteria (for publishing the experiment + model):
  1. Phase 2 complete (hope_final.pth exists, Q&A format learned)
  2. Factual cloze top-5 hit rate >= 60%
  3. Q&A samples actually answer their questions (eyeball check)
  4. Val perplexity <= 30
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F

import train_hope as th

STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_stats.jsonl")

# (cue, expected continuation) — expected's FIRST token is scored
FACTUAL_PROBES = [
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("Romeo and Juliet was written by", " Shakespeare"),
    ("Albert Einstein was a famous", " physicist"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("The sun rises in the", " east"),
    ("Water boils at 100 degrees", " Celsius"),
    ("The chemical formula for water is H2", "O"),
    ("World War II ended in the year", " 1945"),
    ("The language spoken in France is", " French"),
    ("The currency of the United States is the", " dollar"),
    ("Mount Everest is the tallest mountain in the", " world"),
    ("The human heart pumps", " blood"),
    ("The Earth orbits around the", " Sun"),
    ("Photosynthesis converts sunlight into chemical", " energy"),
    ("The first person to walk on the Moon was Neil", " Armstrong"),
    ("Ice turns into water when it starts to", " melt"),
]

QA_PROBES = [
    "Question: Why is the sky blue?\nAnswer:",
    "Question: What is the capital of France?\nAnswer:",
    "Question: Who wrote Romeo and Juliet?\nAnswer:",
    "Question: What is water made of?\nAnswer:",
    "Question: How many legs does a dog have?\nAnswer:",
]

GEN_PROMPTS = [
    "The history of mathematics begins with",
    "Albert Einstein was born in",
    "The French Revolution began in",
]

CHECKPOINTS = ["hope_foundation_best.pth", "hope_foundation.pth",
               "hope_final_best.pth", "hope_final.pth"]


def load_probed_steps():
    seen = {}
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen[rec["source"]] = rec["step"]
                except Exception:
                    pass
    return seen


def load_checkpoint(path):
    """Copy-then-load with retries: avoids racing the trainer's writes."""
    tmp = "/tmp/hope_probe_copy.pth"
    last_err = None
    for attempt in range(4):
        shutil.copy(path, tmp)
        try:
            ckpt = torch.load(tmp, map_location="cpu", weights_only=False)
            model = th.HOPE(th.VOCAB_SIZE, 512, 16,
                            cms_tiers=ckpt.get("cms_tiers") or [[8, 1], [5, 4], [3, 16]])
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.eval()
            return model, ckpt
        except Exception as e:  # torn read — trainer was mid-save
            last_err = e
            time.sleep(4)
    raise RuntimeError(f"could not load {path}: {last_err}")


def next_token_probs(model, cue):
    ids = th.TOKENIZER.encode(cue, return_tensors="pt")
    with torch.no_grad():
        logits, _ = model(ids)
    return F.softmax(logits[0, -1].float(), dim=-1)


def rank_and_prob(probs, token_id):
    p = probs[token_id].item()
    rank = int((probs > probs[token_id]).sum().item()) + 1
    return rank, p


def generate(model, cue, n_tokens=48, temp=0.7, topk=40, seed=42):
    torch.manual_seed(seed)
    ids = th.TOKENIZER.encode(cue, return_tensors="pt")
    with torch.no_grad():
        logits, state = model(ids)
        out = []
        for _ in range(n_tokens):
            p = F.softmax(logits[0, -1].float() / temp, dim=-1)
            v, i = torch.topk(p, topk)
            nxt = i[torch.multinomial(v / v.sum(), 1)]
            if nxt.item() == th.EOS_TOKEN_ID:
                break
            out.append(nxt.item())
            logits, state = model(nxt.view(1, 1), state=state)
    return th.TOKENIZER.decode(out, skip_special_tokens=True)


def probe(path, force=False):
    seen = load_probed_steps()
    model, ckpt = load_checkpoint(path)
    step = ckpt.get("step", -1)
    if not force and seen.get(path) == step:
        print(f"skip {path}: step {step} already probed")
        return None

    t0 = time.time()
    factual = []
    for cue, expected in FACTUAL_PROBES:
        expected_id = th.TOKENIZER.encode(expected, add_special_tokens=False)[0]
        probs = next_token_probs(model, cue)
        rank, p = rank_and_prob(probs, expected_id)
        top5 = [th.TOKENIZER.decode([i]).strip() for i in torch.topk(probs, 5).indices]
        factual.append({"cue": cue, "expected": expected.strip(),
                        "rank": rank, "prob": round(p, 4), "top5": top5})
    hits = sum(1 for r in factual if r["rank"] <= 5)
    mrr = sum(1.0 / r["rank"] for r in factual) / len(factual)

    qa_samples = [generate(model, q, n_tokens=40) for q in QA_PROBES]
    gen_samples = {g: generate(model, g, n_tokens=60) for g in GEN_PROMPTS}

    val_loss = ckpt.get("best_val_loss")
    rec = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": path,
        "step": step,
        "val_loss": round(val_loss, 4) if val_loss else None,
        "val_ppl": round(float(torch.tensor(val_loss).exp()), 1) if val_loss else None,
        "factual_top5_hit_rate": round(hits / len(factual), 3),
        "factual_mrr": round(mrr, 3),
        "factual": factual,
        "qa_samples": qa_samples,
        "gen_samples": gen_samples,
        "probe_seconds": round(time.time() - t0, 1),
    }
    with open(STATS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"\n=== {path} @ step {step} | val_loss {rec['val_loss']} | ppl {rec['val_ppl']} ===")
    print(f"factual top-5 hit rate: {rec['factual_top5_hit_rate']:.0%}  MRR: {rec['factual_mrr']:.3f}  ({rec['probe_seconds']}s on CPU)")
    for r in factual:
        mark = "HIT " if r["rank"] <= 5 else f"#{r['rank']:<4}"
        print(f"  [{mark}] {r['cue']!r} -> expected {r['expected']!r} (p={r['prob']:.3f}); top5={r['top5']}")
    print("\nQ&A samples:")
    for q, a in zip(QA_PROBES, qa_samples):
        print(f"  {q.splitlines()[0]}\n    A: {a[:160]}")
    print("\nGeneration samples:")
    for g, s in gen_samples.items():
        print(f"  {g!r}\n    >>> {s[:220]}")
    return rec


def report():
    if not os.path.exists(STATS_FILE):
        print("no stats yet")
        return
    print(f"{'time':<20}{'source':<28}{'step':>7}{'val_ppl':>9}{'top5%':>7}{'MRR':>7}")
    with open(STATS_FILE) as f:
        for line in f:
            r = json.loads(line)
            print(f"{r['time']:<20}{r['source']:<28}{r['step']:>7}"
                  f"{str(r['val_ppl']):>9}{r['factual_top5_hit_rate']*100:>6.0f}%{r['factual_mrr']:>7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        report()
        sys.exit(0)

    found = False
    for cp in CHECKPOINTS:
        if os.path.exists(cp):
            found = True
            try:
                probe(cp, force=args.force)
            except Exception as e:
                print(f"probe failed for {cp}: {e}")
    if not found:
        print("no checkpoints found yet")
