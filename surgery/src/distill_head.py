#!/usr/bin/env python3
"""Closed-form head transplant: solve HOPE's output layer so that its softmax
reproduces GPT-2's next-token distribution, as far as HOPE's own representation
can express it.

Why this is exact rather than approximate: softmax(log p) == p, because
softmax_i(log p) = p_i / sum_j p_j = p_i. So the least-squares target is the
teacher's LOG-PROBABILITIES, and a perfect fit reproduces the teacher's distribution
with no temperature fudge.

    min_W  || [h_t 1] W - log p_gpt2(·|x_<=t) ||^2   over positions t of a corpus
    W = (X^T X + ridge*I)^-1 X^T Y          <- one 513x513 solve, no gradients

`h_t` is HOPE's own final hidden state (post-CMS, pre-head), so nothing outside
the head changes and the fast memory / nested tiers are left exactly as trained.
Corpus: wikitext-2 TRAIN only — test/val stay untouched as held-out metrics.
"""
import functools
import hashlib
import os

import torch
import torch.nn.functional as F

import hope_lib as hl

DATA = hl.DATA


def state_fingerprint(state):
    m = hashlib.md5()
    for k in sorted(state):
        t = state[k]
        m.update(k.encode())
        m.update(bytes(str(tuple(t.shape)), "utf8"))
        m.update(t.reshape(-1)[:64].float().cpu().numpy().tobytes())
        m.update(str(round(float(t.float().sum()), 4)).encode())
    return m.hexdigest()[:12]


@functools.lru_cache(maxsize=4)
def corpus_windows(split="train", n_windows=300, seq=512, qa_frac=0.0, seed=7):
    """Deterministic packed windows. qa_frac>0 mixes in `Question:/Answer:/Reasoning:`
    formatted items so the fitted head keeps the release's Q&A behaviour."""
    wins = hl.wikitext_windows(split, int(round(n_windows * (1 - qa_frac))) or 1, seq)
    if qa_frac > 0:
        from datasets import load_dataset
        ds = load_dataset("obekt/obekt-question-answer-reasoning-micro-v0.1", split="train")
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)
        rows = []
        need = int(round(n_windows * qa_frac))
        for j in idx.tolist():
            item = ds[j]
            txt = hl.th.format_qa_item(item, ["question", "answer", "reasoning"])
            ids = hl.tok(txt)
            if len(ids) < 16:
                continue
            rows.extend(ids)
            if len(rows) >= need * seq:
                break
        qa = torch.tensor(rows[:need * seq], dtype=torch.long).view(need, seq)
        wins = torch.cat([wins, qa], 0)
    return wins


def teacher_and_hidden(state, tiers, n_windows=300, seq=512, stride=1,
                       clip=-25.0, device="mps", split="train", qa_frac=0.0, ttemp=1.0,
                       mixbase=0.0, wmode="none", batch=2):
    """Returns (A, B, n) with A = X^T X (513x513), B = X^T Y (513x50257).

    mixbase>0 fits the mixture (1-m)*teacher + m*model-instead-of-only-the-teacher:
    because softmax(log q) == q, a perfect fit reproduces that mixture, so the
    head keeps whatever the release's own head did well (in-context behaviour)
    while borrowing the teacher's calibration.
    """
    fp = state_fingerprint(state)
    tag = (f"distill_{fp}_{split}_{n_windows}x{seq}_s{stride}_c{clip}_q{qa_frac}"
           f"_t{ttemp}_m{mixbase}_w{wmode}_b{batch}")
    cache = os.path.join(DATA, tag + ".pt")
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=True)
        return d["A"].double(), d["B"].double(), int(d["n"]), {"cached": True, "fp": fp}

    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    model = hl.model_from_state(state, tiers, device)

    wins = corpus_windows(split, n_windows, seq, qa_frac)
    # Normal equations are accumulated on-device in fp32 (a 513x50257 matrix is
    # 103 MB; shuttling every position slice to the CPU instead would move tens
    # of GB). The solve at the end is done in float64 on the CPU.
    A = torch.zeros(513, 513, dtype=torch.float32, device=device)
    B = torch.zeros(513, hl.VOCAB, dtype=torch.float32, device=device)
    n = 0
    # unigram prior over the fit corpus, for rare-target weighting
    flat = wins.reshape(-1)
    cnt = torch.bincount(flat, minlength=hl.VOCAB).float().clamp_min(1.0)
    uni = (cnt / cnt.sum()).to(device)
    with torch.no_grad():
        for i in range(0, wins.shape[0], batch):
            xb = wins[i:i + batch].to(device)                   # [B, seq]
            # --- HOPE hidden state at every position (pre-head)
            he = model.embedding(xb)
            fo, _ = model.fast_memory(he)
            h = model.norm_fast(he + fo)
            for l in model.cms_layers:
                h = l(h)
            # --- GPT-2 log-probs at every position
            tl = teacher(xb).logits[:, :-1].float()             # predicts x[:,1:]
            lp = F.log_softmax(tl / ttemp, dim=-1).clamp_min(clip)   # [2, seq-1, V]
            if mixbase > 0:
                own = F.log_softmax(model.head(h[:, :-1]).float(), dim=-1).clamp_min(clip)
                lp = ((1 - mixbase) * lp.exp() + mixbase * own.exp()).clamp_min(1e-30).log()
                del own
            # align: sample t uses h[:, t] to predict x[:, t+1]
            hs = h[:, :-1].float()
            tgt = xb[:, 1:]
            keep = torch.ones(hs.shape[:2], dtype=torch.bool, device=device)
            keep &= (tgt != hl.EOS)
            if stride > 1:
                pos = torch.arange(hs.shape[1], device=device)
                keep &= (pos % stride == 0).unsqueeze(0)
            kmask = keep
            nnz = int(kmask.sum())
            X = torch.cat([hs[kmask], torch.ones(nnz, 1, device=device)], 1).float()
            Y = lp[kmask].float()
            # Position weights: plain LS is dominated by trivial positions (the
            # unigram prior, punctuation), which is exactly what buries the
            # context-dependent directions. Renormalised to mean 1 so `ridge`
            # keeps the same meaning for every wmode. Still one linear solve.
            if wmode == "none":
                w = torch.ones(nnz, 1, device=device)
            else:
                if wmode == "surprisal":
                    wp = 1.0 - lp.exp().max(-1).values
                elif wmode == "surprisal2":
                    wp = (1.0 - lp.exp().max(-1).values) ** 2
                elif wmode == "raretgt":
                    wp = (1.0 / (uni[tgt] + 1e-9)).pow(0.25).clamp(max=30.0).log1p()
                else:
                    raise ValueError(wmode)
                w = wp[kmask].unsqueeze(1).float()
                w = w / w.mean().clamp_min(1e-12)
            Xw = X * w
            A += Xw.T @ X
            B += Xw.T @ Y
            n += nnz
            del tl, lp, hs, X, Y, Xw, w
    Ac, Bc = A.cpu(), B.cpu()              # CPU: the solve runs in float64
    torch.save({"A": Ac, "B": Bc, "n": n}, cache)
    del teacher, model
    return Ac.double(), Bc.double(), n, {"cached": False, "fp": state_fingerprint(state)}


def solve_head(A, B, ridge=1.0, ridge_diag_scale=None):
    """W = (A + ridge*I)^-1 B  ->  (head.weight, head.bias)."""
    d = A.shape[0]
    reg = ridge * (A.diagonal().mean() if ridge_diag_scale is None
                   else ridge_diag_scale) * torch.eye(d, dtype=A.dtype)
    W = torch.linalg.solve(A + reg, B)                       # (513, V)
    return W[:d - 1].T.contiguous(), W[d - 1].contiguous()
