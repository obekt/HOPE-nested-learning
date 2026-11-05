#!/usr/bin/env python3
"""Cheap go/no-go for the closed-form head transplant."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as es  # noqa: E402
import hope_lib as hl  # noqa: E402
import surgeries as sg  # noqa: E402

base, ck = hl.state_dict_of(os.path.join(hl.REPO, "surgery/downloads/hope_foundation_best.pth"))
print("baseline:", es.lm_ppl(hl.model_from_state(base, ck["cms_tiers"], "mps"), "mps",
                             n_windows=16), flush=True)

# GPT-2 teacher's own CE on the same windows, for reference
from transformers import AutoModelForCausalLM  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
t = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).to("mps").eval()
wins = hl.wikitext_windows("test", 16, 512).to("mps")
tot = cnt = 0
with torch.no_grad():
    for i in range(0, 16, 2):
        xb = wins[i:i + 2]
        lg = t(xb).logits[:, :-1]
        nll = -Fn.log_softmax(lg.float(), -1).gather(2, xb[:, 1:].unsqueeze(2)).squeeze(2)
        m = xb[:, 1:] != hl.EOS
        tot += (nll * m).sum().item(); cnt += m.sum().item()
print(f"GPT-2 teacher on same windows: ce {tot/cnt:.3f} ppl {torch.tensor(tot/cnt).exp():.1f}",
      flush=True)
del t

for nw, stride, ridge in [(150, 2, 0.003), (150, 2, 0.03)]:
    t0 = time.time()
    st = sg.distill_head(base, n_windows=nw, seq=512, stride=stride, ridge=ridge,
                         alpha=1.0, device="mps")
    meta = st.pop("_distill_meta")
    m = hl.model_from_state(st, ck["cms_tiers"], "mps")
    r = es.lm_ppl(m, "mps", n_windows=16)
    r2 = es.lm_ppl(m, "mps", n_windows=16, split="validation")
    print(f"distill nw={nw} stride={stride} ridge={ridge}: fit_pos={meta['n_positions']} "
          f"{meta['cached']} | TEST {r} | VAL {r2} | head_norm {meta['old_head_norm_mean']:.2f}"
          f" -> {meta['new_head_norm_mean']:.2f} [{time.time()-t0:.0f}s]", flush=True)
    cm = es.cloze_metrics(m, "mps", es.SPEC["cloze_repo_A"] + es.SPEC["cloze_new_A"])
    print("   cloze_A:", {k: v for k, v in cm.items() if k != "items"}, flush=True)
