#!/usr/bin/env python3
"""Closed-form weight edits. No gradients, no optimizer steps, no data fitting
beyond linear algebra on the checkpoint itself (+ one mean-activation vector).

Every op: f(state, **params) -> new state (a copy; base state is never mutated).
"""
import functools
import math

import torch
import torch.nn.functional as F

import hope_lib as hl

INIT_E = 0.02 * math.sqrt(512)          # nn.Embedding normal(0, 0.02) row norm


def copy(state):
    return {k: v.clone() for k, v in state.items()}


# ------------------------------------------------------------------ depth / CMS
def scale_cms_mlp(state, layers, factor):
    """Scale each named CMS block's MLP output by `factor`.

    factor=0 -> block becomes LayerNorm(x): the block contributes nothing but
    renormalization. factor=1 -> untouched. Intermediates interpolate.
    """
    s = copy(state)
    for l in layers:
        for p in (f"cms_layers.{l}.net.0.weight", f"cms_layers.{l}.net.0.bias",
                  f"cms_layers.{l}.net.2.weight", f"cms_layers.{l}.net.2.bias"):
            s[p] = s[p] * factor
    return s


def dup_cms(state, src, dst):
    """Overwrite `dst` blocks with copies of `src` blocks (block-by-block)."""
    s = copy(state)
    for d, s_ in zip(dst, src):
        for suffix in ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias",
                       "norm.weight", "norm.bias"):
            s[f"cms_layers.{d}.{suffix}"] = s[f"cms_layers.{s_}.{suffix}"].clone()
    return s


# ------------------------------------------------------------------- task vector
def task_vector(state, other, lam, exclude=()):
    """state <- state + lam * (other - state), i.e. lam=0 keeps state, 1 -> other."""
    exclude = tuple(exclude) if not isinstance(exclude, str) else (exclude,)
    s = copy(state)
    for k in s:
        if k.startswith(exclude):
            continue
        s[k] = s[k] + lam * (other[k].float() - s[k].float())
    return s


# ------------------------------------------------------------------ vocab rows
def _dead_mask(state, thresh):
    ne = state["embedding.weight"].norm(dim=1)
    return ne <= thresh * INIT_E


def emb_scale_untrained(state, factor, thresh=1.08, floor_to_zero=False):
    """Shrink the embedding rows that never moved off their init norm."""
    s = copy(state)
    dead = _dead_mask(state, thresh)
    E = s["embedding.weight"]
    if floor_to_zero:
        E[dead] = 0.0
    else:
        E[dead] = E[dead] * factor
    s["embedding.weight"] = E
    return s


@functools.lru_cache(maxsize=1)
def gpt2_wte():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("gpt2", "pytorch_model.bin")
    return torch.load(p, map_location="cpu", weights_only=True)["wte.weight"].float()


@functools.lru_cache(maxsize=1)
def gpt2_h():
    """GPT-2's tied output embedding == its wte; kept separate for clarity."""
    return gpt2_wte()


def _ridge_fit(X, Y, lam):
    """Closed-form ridge with intercept. X:(n,di) Y:(n,dout) -> W:(di+1,dout)."""
    Xa = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    Xt = Xa.transpose(0, 1)
    A = Xt @ Xa
    A.diagonal().add_(lam * X.shape[0] / 1000.0)
    return torch.linalg.solve(A, Xt @ Y)


def emb_impute(state, source="gpt2", ridge=30.0, thresh=1.08, norm_mode="median_dead",
               blend=1.0, base="surgery/downloads/hope_foundation_best.pth"):
    """Replace never-trained embedding rows with a closed-form prediction from a
    semantic reference (GPT-2's token embedding, same 50,257 ids), mapped into
    HOPE's own embedding space by ridge regression fitted on the rows HOPE did
    train. No gradient descent: one 769x769 linear solve."""
    s = copy(state)
    E = s["embedding.weight"].float()
    H = s["head.weight"].float()
    ne = E.norm(dim=1)
    alive = ne > 1.05 * INIT_E                      # fit set: rows that moved
    dead = _dead_mask(state, thresh)                # impute set
    G = gpt2_wte()
    X_alive = G if source == "gpt2" else torch.cat([G, H], 1)
    X_alive = X_alive[alive]
    W = _ridge_fit(X_alive, E[alive], ridge)
    X_all = G if source == "gpt2" else torch.cat([G, s["head.weight"].float()], 1)
    pred = torch.cat([X_all, torch.ones(X_all.shape[0], 1)], 1) @ W
    if norm_mode == "median_dead":
        tgt = ne[dead].median()
        pred = pred * (tgt / pred.norm(dim=1, keepdim=True).clamp_min(1e-8))
    elif norm_mode == "keep":
        pd = pred[dead]
        pred = pred.clone()
        pred[dead] = pd * (ne[dead].unsqueeze(1) / pd.norm(dim=1, keepdim=True).clamp_min(1e-8))
    E_new = E.clone()
    E_new[dead] = (1 - blend) * E[dead] + blend * pred[dead]
    s["embedding.weight"] = E_new
    s["_impute_meta"] = {"n_alive": int(alive.sum()), "n_dead": int(dead.sum()),
                         "cos_pred_vs_dead": float(F.cosine_similarity(
                             pred[dead], E[dead], dim=1).mean())}
    return s


def head_equalize(state, strength=1.0, base="surgery/downloads/hope_foundation_best.pth"):
    """Equalize output-row norms (rare tokens currently have ~2.4x the norm of
    frequent ones, so their logits have far higher context variance), while
    compensating head.bias so the MEAN logit of every token is preserved.

    b_i' = b_i + (1 - c_i) * (h_bar . w_i),  w_i' = c_i * w_i,  c_i = (nbar/n_i)^strength
    """
    s = copy(state)
    H = s["head.weight"].float()
    n = H.norm(dim=1).clamp_min(1e-8)
    nbar = torch.exp(torch.log(n).mean())                  # geometric mean norm
    c = (nbar / n) ** strength
    s["head.weight"] = H * c.unsqueeze(1)
    hbar = mean_final_hidden(base)
    s["head.bias"] = s["head.bias"].float() + (1 - c) * (H @ hbar)
    s["_head_eq_meta"] = {"n_ratio_p05_p95": [float(n.quantile(0.05) / nbar),
                                              float(n.quantile(0.95) / nbar)]}
    return s


@functools.lru_cache(maxsize=4)
def mean_final_hidden(path):
    """Average post-CMS hidden vector over held-out wiki tokens (fp32, CPU)."""
    state, ck = hl.state_dict_of(path)
    model = hl.model_from_state(state, ck.get("cms_tiers"), "cpu")
    wins = hl.wikitext_windows("test", 16, 512)
    acc = torch.zeros(512)
    with torch.no_grad():
        for i in range(0, wins.shape[0], 4):
            x = model.embedding(wins[i:i + 4])
            f, _ = model.fast_memory(x)
            h = model.norm_fast(x + f)
            for l in model.cms_layers:
                h = l(h)
            acc += h.reshape(-1, 512).mean(0)
    return (acc / math.ceil(wins.shape[0] / 4)).float()


def head_bias_scale(state, factor):
    s = copy(state)
    s["head.bias"] = s["head.bias"].float() * factor
    return s


def logit_scale(state, scale):
    """Pure temperature control: multiply every logit. Behaviourally identical
    to changing sampling temperature, so it must never count as a capability gain."""
    s = copy(state)
    s["head.weight"] = s["head.weight"].float() * scale
    s["head.bias"] = s["head.bias"].float() * scale
    return s


def gate_shift(state, d_alpha=0.0, d_beta=0.0):
    """Shift the fast-memory forget gate (alpha=sigmoid(gw.x+4+bias+d_alpha)) and
    the inner learning rate (beta=sigmoid(gw.x-2+bias+d_beta))."""
    s = copy(state)
    s["fast_memory.gate_alpha.bias"] = s["fast_memory.gate_alpha.bias"].float() + d_alpha
    s["fast_memory.gate_beta.bias"] = s["fast_memory.gate_beta.bias"] + d_beta
    return s


def fast_out_scale(state, factor):
    s = copy(state)
    s["fast_memory.proj_out.weight"] = s["fast_memory.proj_out.weight"].float() * factor
    s["fast_memory.proj_out.bias"] = s["fast_memory.proj_out.bias"].float() * factor
    return s


def norm_fast_rescale(state, factor):
    s = copy(state)
    s["norm_fast.weight"] = s["norm_fast.weight"].float() * factor
    return s


DEFAULT_TIERS = [[8, 1], [5, 4], [3, 16]]


def emb_from_head(state, mode="unit", target="median_alive", blend=1.0):
    """Rewrite the INPUT embedding from the model's own OUTPUT rows.

    Diagnostics: 61% of embedding rows are still at random init, while every head
    row grew 4-35x (the head is the best-trained matrix in the model) and is 63%
    linearly explained by GPT-2's embedding — so H carries far more token semantics
    than E does. This gives every token an input vector in that semantic geometry
    and lets the closed-form head refit adapt to it.
    """
    s = copy(state)
    H, E = s["head.weight"].float(), s["embedding.weight"].float()
    ne = E.norm(dim=1)
    alive = ne > 1.05 * INIT_E
    tgt_norm = {"median_alive": ne[alive].median(),
                 "init": torch.tensor(INIT_E),
                 "mean_all": ne.mean()}[target]
    U = H / H.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if mode == "center":
        U = (H - H.mean(0, keepdim=True))
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-8)
    new_E = U * tgt_norm
    s["embedding.weight"] = (1 - blend) * E + blend * new_E
    return s


def emb_overwrite_g2(state, ridge=30.0, norm_target="median_alive", blend=1.0):
    """Every embedding row <- ridge-map prediction from GPT-2's same-id token
    vector (fitted on the rows HOPE did train). Unlike emb_impute, this replaces
    the whole matrix, so it is only sensible when the head is re-solved after."""
    s = copy(state)
    E, H = s["embedding.weight"].float(), s["head.weight"].float()
    ne = E.norm(dim=1)
    alive = ne > 1.05 * INIT_E
    G = gpt2_wte()
    W = _ridge_fit(G[alive], E[alive], ridge)
    pred = torch.cat([G, torch.ones(G.shape[0], 1)], 1) @ W
    tgt = {"median_alive": ne[alive].median(), "init": torch.tensor(INIT_E),
           "mean_all": ne.mean()}[norm_target]
    pred = pred * (tgt / pred.norm(dim=1, keepdim=True).clamp_min(1e-8))
    s["embedding.weight"] = (1 - blend) * E + blend * pred
    return s


@functools.lru_cache(maxsize=2)
def bigram_topk(n_windows=1000, seq=512, k=48, split="train", qa_frac=0.0):
    """(ids, probs) [V, k] of the most likely successors of every vocabulary
    token, counted in the fit corpus. Pure counting — no fitting, no gradients."""
    import distill_head as dh
    wins = dh.corpus_windows(split, n_windows, seq, qa_frac)
    a = wins[:, :-1].reshape(-1)
    b = wins[:, 1:].reshape(-1)
    sel = (b != hl.EOS) & (a != hl.EOS)
    a, b = a[sel].long(), b[sel].long()
    key = a * hl.VOCAB + b
    uk, cnt = torch.unique(key, return_counts=True)
    aa, bb, cc = uk // hl.VOCAB, uk % hl.VOCAB, cnt.long()
    big = int(cc.max()) + 1                                # torch has no lexsort: sort
    order = torch.argsort(aa * (2 * big) + (big - cc))     # on one composite int64 key
    aa, bb, cc = aa[order], bb[order], cc[order]
    first = torch.cat([torch.ones(1, dtype=torch.bool), aa[1:] != aa[:-1]])
    total = torch.zeros(hl.VOCAB, dtype=torch.float64).index_add_(
        0, aa, cc.double())                                 # occurrences of token a
    grp = first.cumsum(0) - 1
    starts = torch.nonzero(first).flatten()
    rank = torch.arange(len(aa)) - starts[grp]
    keep = rank < k
    ids = torch.zeros(hl.VOCAB, k, dtype=torch.long)
    prb = torch.zeros(hl.VOCAB, k, dtype=torch.float64)
    aa_k = aa[keep]
    ids[aa_k, rank[keep]] = bb[keep]
    prb[aa_k, rank[keep]] = cc[keep].double() / total[aa_k].clamp_min(1)
    return ids, prb, total


def succ_vmap(state, vscale=1.0, ridge=1.0, k=48, n_windows=1000, floor=1e-6,
              split="train", qa_frac=0.0, fit_thresh=2):
    """Rewrite the fast memory's VALUE projection so the vector written for token
    a is the pre-image, under the model's *current* head, of a's observed successor
    distribution.

    After this, the delta-rule memory holds "what usually follows each context
    token" instead of a random projection of the token, and a read over the
    context becomes a bag of successor votes — i.e. the memory is turned into a
    lookup table by closed-form linear algebra (pseudo-inverse + one 513x513
    ridge solve). `vscale` sets its norm; nothing here is trained.
    """
    s = copy(state)
    W = s["head.weight"].float()                       # (V, 512) current output layer
    G = W.transpose(0, 1) @ W                          # (512, 512)
    P = torch.linalg.solve(G + ridge * G.diagonal().mean() * torch.eye(512),
                           W.transpose(0, 1))          # (512, V) = pinv(W)
    ids, prb, total = bigram_topk(n_windows, 512, k, split, qa_frac)
    ok = (total >= fit_thresh)                         # tokens seen often enough
    logd = (prb.clamp_min(1e-12).log() - math.log(floor)) * (prb > 0).double()
    n_ok = int(ok.sum())
    ok_idx = torch.nonzero(ok).flatten()
    coef_all = logd[ok].float()                          # (n_ok, k)
    ids_all = ids[ok]                                    # (n_ok, k)
    parts = []
    CH = 1024                                            # chunk to bound memory
    for j in range(0, n_ok, CH):
        c, d = coef_all[j:j + CH], ids_all[j:j + CH]
        m = c.shape[0]
        Psel = P[:, d.reshape(-1)]                       # (512, m*k)
        v = (Psel * c.reshape(-1).unsqueeze(0)).reshape(512, m, k).sum(2).T
        parts.append(v)
        del Psel
    v = torch.cat(parts, 0)                              # (n_ok, 512)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
    v = v.float() * vscale
    E = s["embedding.weight"].float()[ok]               # (n_ok, 512)
    wt = torch.log1p(total[ok].float()).unsqueeze(1)
    Wv = _ridge_fit(E * wt, v * wt, 1.0)               # (513, 512)
    s["fast_memory.proj_v.weight"] = Wv[:512].T.contiguous()
    s["fast_memory.proj_v.bias"] = Wv[512].clone()
    s["_succ_meta"] = {"n_fit_tokens": int(ok.sum()), "vscale": vscale}
    return s


def distill_head(state, n_windows=300, seq=512, stride=1, clip=-25.0, ridge=0.01,
                 alpha=1.0, split="train", tiers=None, device="mps", qa_frac=0.0,
                 ttemp=1.0, mixbase=0.0, wmode="none", batch=2):
    """Replace the output layer with the closed-form least-squares solution that
    makes HOPE's softmax approximate GPT-2's next-token distribution."""
    import distill_head as dh
    A, B, n, meta = dh.teacher_and_hidden(
        state, tiers or DEFAULT_TIERS, n_windows, seq, stride, clip, device, split,
        qa_frac, ttemp, mixbase, wmode, batch)
    W, b = dh.solve_head(A, B, ridge)
    s = copy(state)
    s["head.weight"] = (1 - alpha) * s["head.weight"].float() + alpha * W
    s["head.bias"] = (1 - alpha) * s["head.bias"].float() + alpha * b
    s["_distill_meta"] = {"n_positions": n, "ridge": ridge, "alpha": alpha,
                         "mixbase": mixbase, "qa_frac": qa_frac, "wmode": wmode,
                         "fingerprint": meta["fp"], "cached": meta["cached"],
                         "new_head_norm_mean": float(W.norm(dim=1).mean()),
                         "old_head_norm_mean": float(
                             state["head.weight"].float().norm(dim=1).mean())}
    return s


def mem_tune(state, gamma_q=1.0, gamma_k=1.0, v_mode="keep", out_mode="keep",
             out_scale=1.0, tie_qk=False):
    """Re-tune the fast memory's read/write geometry (still no gradients).

    Diagnostics say proj_q/k/v/out sit within ~15% of their random init while the
    two gates moved 4-7x, so associative retrieval never became sharp: q_t . k_s is
    a random overlap for every pair, including s == t.

      gamma_q / gamma_k < 1 shrink the token-dependent part of query/key while the
        bias stays, so keys collapse toward a shared direction and the read becomes
        a positively-weighted aggregate of past values (a context bag) instead of a
        random-sign mixture.
      tie_qk: q <- k (self-matching read: retrieve what was written at the best
        key match, the primitive a delta-rule memory is actually built for).
      v_mode "identity": values carry the token embedding itself (what token was
        here), "head": values carry the OLD output row (closed-form E->H fit).
      out_mode "identity": fast_out lands back in embedding space, which the
        trained CMS knows how to read.
    """
    s = copy(state)
    fm = "fast_memory."
    s[fm + "proj_q.weight"] = s[fm + "proj_q.weight"].float() * gamma_q
    s[fm + "proj_k.weight"] = s[fm + "proj_k.weight"].float() * gamma_k
    if tie_qk:
        s[fm + "proj_q.weight"] = s[fm + "proj_k.weight"].clone()
        s[fm + "proj_q.bias"] = s[fm + "proj_k.bias"].clone()
    E, H = s["embedding.weight"].float(), s["head.weight"].float()
    if v_mode == "identity":
        s[fm + "proj_v.weight"] = torch.eye(512)
        s[fm + "proj_v.bias"] = torch.zeros(512)
    elif v_mode == "head":
        Ea = torch.cat([E, torch.ones(E.shape[0], 1)], 1)
        W = _ridge_fit(Ea, H, 10.0)                # (513, 512)
        s[fm + "proj_v.weight"] = W[:512].T.contiguous()
        s[fm + "proj_v.bias"] = W[512].clone()
    if out_mode == "identity":
        s[fm + "proj_out.weight"] = out_scale * torch.eye(512)
        s[fm + "proj_out.bias"] = torch.zeros(512)
    elif out_mode == "blend":
        I = torch.eye(512)
        s[fm + "proj_out.weight"] = out_scale * (
            0.5 * s[fm + "proj_out.weight"].float() + 0.5 * I)
    return s


OPS = {
    "identity": lambda st, **kw: copy(st),
    "distill_head": distill_head,
    "mem_tune": mem_tune,
    "succ_vmap": succ_vmap,
    "emb_from_head": emb_from_head,
    "emb_overwrite_g2": emb_overwrite_g2,
    "scale_cms_mlp": scale_cms_mlp,
    "dup_cms": dup_cms,
    "task_vector": task_vector,
    "emb_scale_untrained": emb_scale_untrained,
    "emb_impute": emb_impute,
    "head_equalize": head_equalize,
    "head_bias_scale": head_bias_scale,
    "logit_scale": logit_scale,
    "gate_shift": gate_shift,
    "fast_out_scale": fast_out_scale,
    "norm_fast_rescale": norm_fast_rescale,
}


def apply_pipeline(base_state, steps, other=None):
    """steps: list of {op, params}. `other` is provided for task_vector steps."""
    st = base_state
    for stp in steps:
        op = OPS[stp["op"]]
        params = dict(stp.get("params", {}))
        if stp["op"] == "task_vector":
            if other is None:
                raise ValueError("task_vector needs an `other` state")
            st = op(st, other, **params)
        else:
            st = op(st, **params)
        st = {k: v for k, v in st.items() if torch.is_tensor(v)}   # drop step metadata
    return st
