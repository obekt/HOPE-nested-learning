"""Shared helpers: load a HOPE checkpoint, stream held-out text, token utils.

Nothing here mutates a checkpoint. Scripts that DO write weights live in this
same dir but call save_surgery() explicitly.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

SRC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SRC))          # .../HOPE-nested-learning
sys.path.insert(0, REPO)
import train_hope as th  # noqa: E402
SRC = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "surgery", "data")
OUT = os.path.join(REPO, "surgery", "out")
os.makedirs(DATA, exist_ok=True)

VOCAB = th.VOCAB_SIZE
EOS = th.EOS_TOKEN_ID


def load_model(path, device="mps", dtype=torch.float32):
    """Strict fp32 loader — surgery effects must not be masked by bf16 rounding."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    tiers = ck.get("cms_tiers") or [[8, 1], [5, 4], [3, 16]]
    return model_from_state(ck["model_state"], tiers, device, dtype), ck


def model_from_state(state, tiers, device="mps", dtype=torch.float32):
    model = th.HOPE(VOCAB, 512, 16, cms_tiers=tiers)
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def state_dict_of(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.clone() for k, v in ck["model_state"].items()}, ck


def save_surgery(base_path, new_state, out_path, note):
    """Write a modified checkpoint, preserving the repo's checkpoint schema."""
    ck = torch.load(base_path, map_location="cpu", weights_only=True)
    ck["model_state"] = {k: v.contiguous() for k, v in new_state.items()}
    ck["surgery"] = note
    torch.save(ck, out_path)
    return out_path


def tok(text):
    """Token ids as a plain list (EOS/PAD never appears in these strings)."""
    return list(th.TOKENIZER.encode(text, add_special_tokens=False))


def tok_t(text):
    return torch.tensor(th.TOKENIZER.encode(text, add_special_tokens=False), dtype=torch.long)


def tok_ids(text):
    return th.TOKENIZER.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------- held-out text
SPLIT_ALIAS = {"val": "validation", "test": "test", "train": "train",
               "validation": "validation"}


def wikitext_windows(split="test", n_windows=64, seq_len=512, cache=True):
    """Fixed, deterministic windows of held-out Wikipedia text (wikitext-2 raw)."""
    split = SPLIT_ALIAS[split]
    cache_path = os.path.join(DATA, f"wt2_{split}_{n_windows}x{seq_len}.pt")
    if cache and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    paras = [x for x in ds["text"] if len(x.strip()) > 80]
    ids = th.TOKENIZER(paras, add_special_tokens=False)["input_ids"]
    flat = []
    for row in ids:
        flat.append(EOS)
        flat.extend(row)
    need = n_windows * seq_len
    while len(flat) < need:            # tiny corpus safety net
        flat.extend(flat)
    wins = torch.tensor(flat[:need], dtype=torch.long).view(n_windows, seq_len)
    if cache:
        torch.save(wins, cache_path)
    return wins


_DEC = {}


def _decode_cache(tid):
    if tid not in _DEC:
        _DEC[tid] = th.TOKENIZER.decode([tid])
    return _DEC[tid]


def copy_eval_pairs(split="val", n_samples=300, seq_len=512, min_gap=24, seed=0):
    """(context_with_cue, context_cue_blanked, target_id) triples.

    Finds a rare token that occurs twice in a window; the second occurrence is
    the target. The control blanks ONLY the first occurrence (length and all
    other positions identical), so the difference isolates in-context copying.
    """
    cache_path = os.path.join(DATA, f"copy_{split}_{n_samples}_{seed}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    wins = wikitext_windows(split, 128, seq_len)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(wins.shape[0], generator=g)
    out = []
    for wi in order:
        w = wins[wi]
        first = {}
        for pos in range(seq_len):
            t = int(w[pos])
            if t not in first:
                first[t] = pos
                continue
            p1 = first[t]
            if t < 5000 or (pos - p1) < min_gap or pos < 8:   # skip function words/punct
                continue
            if not any(c.isalpha() for c in _decode_cache(t)):
                continue
            c_with = w[:pos].clone()
            c_blank = w[:pos].clone()
            c_blank[p1] = 262              # ' the' — filler, keeps length fixed
            out.append({"win": int(wi), "p1": p1, "p2": pos, "target": t,
                        "with": c_with, "blank": c_blank})
            first[t] = pos                 # next pair uses this as the cue
            if len(out) >= n_samples:
                break
        if len(out) >= n_samples:
            break
    if cache_path:
        torch.save(out, cache_path)
    return out


def rank_of(logits_row, target_id):
    """1-based rank of target_id under the distribution row (deterministic)."""
    p = logits_row[target_id].item()
    return int((logits_row > p).sum().item()) + 1


def _as_row(ids):
    t = ids if torch.is_tensor(ids) else torch.tensor(list(ids), dtype=torch.long)
    return t.view(1, -1)


@torch.no_grad()
def last_logits(model, ids_1d, device):
    x = _as_row(ids_1d).to(device)
    logits, _ = model(x)
    return logits[0, -1].float().cpu()


@torch.no_grad()
def batch_logits(model, ids_2d, device):
    logits, _ = model(ids_2d.to(device))
    return logits.float().cpu()


@torch.no_grad()
def next_token_logprobs(model, ids_1d, device):
    """log P(x_t | x_<t) for every position of the sequence (T-1 values)."""
    x = _as_row(ids_1d).to(device)
    logits, _ = model(x)
    lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    return lp.gather(1, x[0, 1:].unsqueeze(1)).squeeze(1).cpu()
