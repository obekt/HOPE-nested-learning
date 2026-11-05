#!/usr/bin/env python3
"""Pre-registered capability suite for HOPE weight surgery.

Usage:
    python3 eval_suite.py --label name=/path/to.pth [--label ...] [--device mps]

Deterministic only (likelihood + greedy): no sampling noise between candidates.
Metric definitions come from eval_spec.json and are frozen.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hope_lib as hl  # noqa: E402

SPEC = json.load(open(os.path.join(hl.SRC, "eval_spec.json")))
OSDIR = os.environ.get("HF_DATASES_OFFLINE", "")


# ------------------------------------------------------------------ lm metrics
@torch.no_grad()
def lm_ppl(model, device, n_windows=64, seq_len=512, split="test"):
    wins = hl.wikitext_windows(split, n_windows, seq_len)
    tot, cnt = 0.0, 0
    for i in range(0, wins.shape[0], 8):
        batch = wins[i:i + 8].to(device)
        logits, _ = model(batch)
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = batch[:, 1:]
        mask = (tgt != hl.EOS)                      # mirrors ignore_index=PAD
        nll = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        tot += (nll * mask).sum().item()
        cnt += mask.sum().item()
    ce = tot / max(cnt, 1)
    return {"ce": round(ce, 4), "ppl": round(math.exp(min(ce, 20)), 3), "tokens": cnt}


def cloze_metrics(model, device, items):
    rows, top1, top5, mrr, ent, p1 = [], 0, 0, 0.0, [], []
    for cue, exp in items:
        tid = hl.tok_ids(exp)[0]
        ids = hl.tok(cue)
        logits = hl.last_logits(model, ids, device)
        probs = F.softmax(logits, dim=-1)
        r = hl.rank_of(logits, tid)
        top1 += r == 1
        top5 += r <= 5
        mrr += 1.0 / r
        ent.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))
        p1.append(float(probs.max()))
        rows.append({"cue": cue, "rank": r, "prob": round(float(probs[tid]), 4)})
    n = len(items)
    return {"top1": round(top1 / n, 4), "top5": round(top5 / n, 4),
            "mrr": round(mrr / n, 4), "n": n,
            "mean_entropy": round(float(np.mean(ent)), 3),
            "mean_top1_prob": round(float(np.mean(p1)), 4), "items": rows}


@torch.no_grad()
def score_choices(model, device, ctx_ids, choice_id_lists):
    """Right-padded batch scoring of [ctx | choice_j] rows.

    Exact for this architecture: PAD==EOS tokens are masked out of the fast
    memory update, and the network has no positional encoding, so trailing pad
    cannot influence the logits we read (all at positions < first pad).
    Returns (mean-logp per choice, choice lengths).
    """
    def _lst(v):
        return v.tolist() if torch.is_tensor(v) else list(v)
    ctx, chs = _lst(ctx_ids), [_lst(c) for c in choice_id_lists]
    lens = [len(c) for c in chs]
    mx = max(lens)
    rows = [ctx + c + [hl.EOS] * (mx - len(c)) for c in chs]
    full = torch.tensor(rows, dtype=torch.long)
    logits, _ = model(full.to(device))
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1).cpu()
    L = len(ctx)
    scores = []
    for j, c in enumerate(chs):
        s = 0.0
        for t, tid in enumerate(c):
            s += float(lp[j, L + t - 1, tid])     # logits at pos p predict p+1
        scores.append(s / max(len(c), 1))
    next_tok = int(logits[0, L - 1].float().argmax())   # greedy token right after ctx
    return scores, lens, next_tok


def mc_suite(model, device, items, name):
    """items: list of (ctx_ids, [choice_ids...], correct_index)

    acc_normalized / acc_raw: choice-style accuracy. first_tok_acc is a
    *generation* metric: does the model's greedy next token after the prompt
    begin the gold answer? (matters for the chat scripts, where there is no
    list of options to rank).
    """
    acc_n, acc_raw, n, wins = 0.0, 0.0, 0, 0
    mc2 = 0.0
    first = 0
    for ctx, choices, gold in items:
        sc, lens, nxt = score_choices(model, device, ctx, choices)
        norm = np.array(sc)                        # mean logp per token
        raw = np.array(sc) * np.array(lens)        # total logp
        wins += int(int(np.argmax(norm)) == gold)
        acc_n += norm[gold]
        acc_raw += int(int(np.argmax(raw)) == gold)
        e = np.exp(norm - norm.max()); mc2 += float(e[gold] / e.sum())
        first += int(list(choices[gold])[0] == nxt)
        n += 1
    return {"n": n, "acc_normalized": round(wins / n, 4),
            "acc_raw": round(acc_raw / n, 4), "first_tok_acc": round(first / n, 4),
            "mc2_like": round(mc2 / n, 4), "mean_logp_gold": round(acc_n / n, 3)}




def _rows(ds, n):
    return ds.select(range(min(n, len(ds)))) if n else ds


def build_mc_items(kind, cfg):
    from datasets import load_dataset
    items = []
    if kind == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=cfg["split"])
        for r in _rows(ds, cfg.get("limit", len(ds))):
            ctx = hl.tok("Question: " + r["question"] + "\nAnswer: ")
            ch = [hl.tok(c) for c in r["choices"]["text"]]
            labels = r["choices"]["label"]
            gold = labels.index(r["answerKey"]) if r["answerKey"] in labels else 0
            items.append((ctx, ch, gold))
    elif kind == "winogrande":
        ds = load_dataset(cfg["name"], cfg["config"], split=cfg["split"])
        for r in _rows(ds, cfg.get("limit")):
            base = r["sentence"].replace("_", " ")
            ch = [hl.tok(o) for o in (r["option1"], r["option2"])]
            ctx = hl.tok(base + " ")
            gold = int(r.get("label", r.get("answer"))) - 1
            if gold < 0:
                continue
            items.append((ctx, ch, gold))
    elif kind == "truthfulqa":
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        for r in _rows(ds, cfg.get("limit")):
            tgt = r["mc1_targets"] if cfg["split"] == "mc1" else r["mc2_targets"]
            ch = [hl.tok(c) for c in tgt["choices"]]
            gold = int(np.argmax(tgt["labels"]))
            items.append((hl.tok(r["question"] + " "), ch, gold))
    elif kind == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split=cfg["split"])
        g = np.random.RandomState(cfg.get("seed", 0))
        idx = g.choice(len(ds), size=min(cfg["limit"], len(ds)), replace=False)
        for i in sorted(idx.tolist()):
            r = ds[int(i)]
            ctx = hl.tok(r["ctx"].strip() + " ")
            ch = [hl.tok(e.strip() + " ") for e in r["endings"]]
            items.append((ctx, ch, int(r["label"])))
    elif kind == "mmlu":
        subj_items = []
        for s in cfg["subjects"]:
            ds = load_dataset("cais/mmlu", "all", split=cfg["split"])
            ds = ds.filter(lambda x: x["subject"] == s, num_proc=None).select(
                range(min(cfg["per_subject"], len(ds))))
            for r in ds:
                ctx = hl.tok(f"Question ({s.replace('_', ' ')}): {r['question']}\nAnswer: ")
                ch = [hl.tok(c) for c in r["choices"]]
                subj_items.append((ctx, ch, int(r["answer"])))
        items = subj_items
    return items


@torch.no_grad()
def copy_gain(model, device, n=300):
    pairs = hl.copy_eval_pairs("val", n)
    acc_w = acc_b = 0
    rank_w = rank_b = 0.0
    for p in pairs:
        lw = hl.last_logits(model, p["with"], device)
        lb = hl.last_logits(model, p["blank"], device)
        rw, rb = hl.rank_of(lw, p["target"]), hl.rank_of(lb, p["target"])
        acc_w += rw == 1; acc_b += rb == 1
        rank_w += rw; rank_b += rb
    m = len(pairs)
    return {"n": m, "acc_with_cue": round(acc_w / m, 4), "acc_blank_cue": round(acc_b / m, 4),
            "gain": round((acc_w - acc_b) / m, 4),
            "mean_rank_with": round(rank_w / m, 2), "mean_rank_blank": round(rank_b / m, 2)}


@torch.no_grad()
def qa_greedy(model, device, prompts, n_tokens=40):
    outs = []
    for pr in prompts:
        x = hl._as_row(hl.tok(pr)).to(device)
        logits, state = model(x)
        gen = []
        for _ in range(n_tokens):
            nxt = int(logits[0, -1].float().argmax())
            gen.append(nxt)
            if state is None:        # stateless reference model: re-run the prefix
                logits = model(torch.tensor([x[0].tolist() + gen], device=device))[0]
            else:
                logits, state = model(torch.tensor([[nxt]], device=device), state=state)
            if nxt == hl.EOS:
                break
        text = hl.th.TOKENIZER.decode(gen, skip_special_tokens=True)
        outs.append(text)
    uniq = len(set(outs))
    return {"samples": outs, "n_unique": uniq}


def evaluate_state(state, device="mps", suites="AB", meta=None, tiers=None):
    model = hl.model_from_state(state, tiers or [[8, 1], [5, 4], [3, 16]], device)
    res = dict(meta or {})
    if "A" in suites:
        res["ppl_wt2_test"] = lm_ppl(model, device, split="test")
        res["cloze_A"] = {k: v for k, v in cloze_metrics(
            model, device, SPEC["cloze_repo_A"] + SPEC["cloze_new_A"]).items() if k != "items"}
        res["cloze_A_items"] = cloze_metrics(model, device, SPEC["cloze_repo_A"] + SPEC["cloze_new_A"])["items"]
        for name, cfg in SPEC["mc_suites_A"].items():
            kind = "winogrande" if "winogrande" in name else name
            cfg = dict(cfg)
            if name == "hellaswag":
                cfg["name"] = "Rowan/hellaswag"
            items = build_mc_items(kind, cfg)
            res["mc_" + name] = mc_suite(model, device, items, name)
        res["copy"] = copy_gain(model, device)
        res["qa_A"] = qa_greedy(model, device, SPEC["qa_prompts_A"])
    if "B" in suites:
        res["ppl_wt2_val"] = lm_ppl(model, device, split="val")
        m = cloze_metrics(model, device, SPEC["cloze_holdout_B"])
        res["cloze_B"] = {k: v for k, v in m.items() if k != "items"}
        res["cloze_B_items"] = m["items"]
        for name, cfg in SPEC["mc_suites_B"].items():
            kind = "mmlu" if name == "mmlu" else "hellaswag"
            cfg = dict(cfg)
            if "hellaswag" in name:
                cfg["name"] = "Rowan/hellaswag"
            items = build_mc_items(kind, cfg)
            res["mc_" + name] = mc_suite(model, device, items, name)
        res["qa_B"] = qa_greedy(model, device, SPEC["qa_prompts_B"])
    del model
    return res


class GPT2Wrap:
    """Same-vocab reference model behind HOPE's (logits, state) interface, so the
    identical suite can be run on GPT-2 small."""

    def __init__(self, device="mps"):
        from transformers import AutoModelForCausalLM
        self.m = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device = device

    def __call__(self, x, state=None, mask=None, last_only=False):
        return self.m(x).logits, None


def evaluate_ref(device="mps"):
    model = GPT2Wrap(device)
    res = {"label": "gpt2_small_reference", "ckpt": "gpt2 (124M, same vocab)"}
    res["ppl_wt2_test"] = lm_ppl(model, device, split="test")
    m = cloze_metrics(model, device, SPEC["cloze_repo_A"] + SPEC["cloze_new_A"])
    res["cloze_A"] = {k: v for k, v in m.items() if k != "items"}
    m = cloze_metrics(model, device, SPEC["cloze_holdout_B"])
    res["cloze_B"] = {k: v for k, v in m.items() if k != "items"}
    for name, cfg in SPEC["mc_suites_A"].items():
        kind = "winogrande" if "winogrande" in name else name
        res["mc_" + name] = mc_suite(model, device, build_mc_items(kind, dict(cfg)), name)
    res["copy"] = copy_gain(model, device)
    res["qa_A"] = qa_greedy(model, device, SPEC["qa_prompts_A"])
    res["ppl_wt2_val"] = lm_ppl(model, device, split="val")
    for name, cfg in SPEC["mc_suites_B"].items():
        kind = "mmlu" if name == "mmlu" else "hellaswag"
        res["mc_" + name] = mc_suite(model, device, build_mc_items(kind, dict(cfg)), name)
    res["qa_B"] = qa_greedy(model, device, SPEC["qa_prompts_B"])
    return res


def evaluate_random(device="mps", seed=0, tiers=None):
    """Freshly initialised HOPE (same architecture, no training) — the floor that
    says how much the release's 48k+6k steps actually bought."""
    torch.manual_seed(seed)
    model = hl.th.HOPE(hl.VOCAB, 512, 16, cms_tiers=tiers or [[8, 1], [5, 4], [3, 16]])
    model = model.to(device=device, dtype=torch.float32).eval()
    res = {"label": f"untrained_hope_seed{seed}", "ckpt": "random init"}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return dict(evaluate_state(state, device, "AB", res, tiers), label=res["label"],
                ckpt=res["ckpt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=False, default=[])
    ap.add_argument("--gpt2ref", action="store_true")
    ap.add_argument("--randomref", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--suites", default="AB")
    ap.add_argument("--out", default=os.path.join(hl.OUT, "eval_results.jsonl"))
    a = ap.parse_args()
    os.makedirs(hl.OUT, exist_ok=True)
    lines = []
    if a.randomref:
        print("[eval] untrained HOPE reference", flush=True)
        rr = evaluate_random(a.device)
        lines.append(rr)
        with open(a.out, "a") as f:
            f.write(json.dumps(rr) + "\n")
        print(json.dumps({k: v for k, v in rr.items()
                          if k not in ("cloze_A_items", "cloze_B_items", "qa_A", "qa_B")},
                         indent=1), flush=True)
    if a.gpt2ref:
        print("[eval] GPT-2 small reference", flush=True)
        rr = evaluate_ref(a.device)
        lines.append(rr)
        with open(a.out, "a") as f:
            f.write(json.dumps(rr) + "\n")
        print(json.dumps({k: v for k, v in rr.items()
                          if k not in ("cloze_A_items", "cloze_B_items", "qa_A", "qa_B")},
                         indent=1), flush=True)
    for lab in a.label:
        name, _, path = lab.partition("=")
        print(f"[eval] {name} <- {path}", flush=True)
        state, ck = hl.state_dict_of(path)
        r = evaluate_state(state, a.device, a.suites,
                           meta={"label": name, "ckpt": os.path.basename(path),
                                 "step": ck.get("step"), "val_loss": ck.get("best_val_loss")},
                           tiers=ck.get("cms_tiers"))
        lines.append(r)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps({k: v for k, v in r.items()
                          if k not in ("cloze_A_items", "cloze_B_items", "qa_A", "qa_B")}, indent=1), flush=True)
    summary = os.path.join(hl.OUT, "last_summary.json")
    json.dump(lines, open(summary, "w"), indent=1)
    print(f"wrote {summary} and appended {a.out}", flush=True)


if __name__ == "__main__":
    main()
