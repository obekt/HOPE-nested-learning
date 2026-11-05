#!/usr/bin/env python3
"""End-to-end proof that an edited checkpoint is a usable release artifact:
loaded through the repo's OWN inference stack (bf16, last-only prefill, compiled
steps, sample_next_token, prompt-format auto-detection), timed, and sampled."""
import json
import os
import sys
import time

import torch

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SRC)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_hope as th  # noqa: E402
import hope_lib as hl  # noqa: E402

path = sys.argv[1]
dev = th.DEVICE
model, ck = th.load_model_for_inference(path, device=dev, dtype=torch.bfloat16)
print(f"loaded {path} strict=True via load_model_for_inference "
      f"({dev}, bf16); step={ck.get('step')} surgery={ck.get('surgery', {}).get('name')}")
qa = th.uses_qa_format(path, ck)
print(f"uses_qa_format -> {qa}")

th.maybe_compile_for_steps(model, verbose=False)


@torch.no_grad()
def run(prompt, n=64, temperature=0.0):
    ids = torch.tensor([th.TOKENIZER.encode(th.wrap_prompt(prompt, qa),
                                            add_special_tokens=False)], device=dev)
    t0 = time.time()
    logits, state = model(ids)
    prefill = (time.time() - t0) * 1000
    out, t1 = [], time.time()
    for _ in range(n):
        nxt = th.sample_next_token(logits, temperature=temperature,
                                   top_k=40, top_p=0.9, repetition_penalty=1.15,
                                   prev_tokens=ids[0].tolist() + out)
        out.append(int(nxt))
        logits, state = model(nxt.to(dev), state=state)
    txt = th.TOKENIZER.decode(out, skip_special_tokens=True)
    return txt, prefill, n / max(time.time() - t1, 1e-6)


print(f"\nprefill ms / generation tok/s on {dev}:")
# bare questions: wrap_prompt() adds the template when qa_format is set, so
# pre-wrapping here would double-wrap and test the wrong thing.
for pr in ["Why is the sky blue?",
           "What is the capital of Japan?",
           "The history of mathematics begins with"]:
    for temp in (0.0, 0.7):
        txt, pf, tps = run(pr, 48, temp)
        print(f"  T={temp:g} [{pr[:40]!r}] prefill {pf:.1f}ms {tps:.0f}tok/s\n"
              f"     >>> {txt[:150].strip()!r}")

full = th.load_model_for_inference(path, device=dev, dtype=torch.float32)[0]
print("\nfp32 strict load OK:", sum(p.numel() for p in full.parameters()), "params")

# The repo documents bf16 as quality-identical for the *released* weights. A
# re-solved head has different row norms, so check the claim again on this file.
wins = hl.wikitext_windows("test", 32, 512)
agree, n, ranks32, ranks16 = 0, 0, [], []
import eval_suite as es  # noqa: E402
for i in range(0, wins.shape[0], 8):
    x = wins[i:i + 8].to(dev)
    with torch.no_grad():
        a = model(x)[0][:, -1].float()
        b = full.to(dev)(x)[0][:, -1].float()
    agree += int((a.argmax(-1) == b.argmax(-1)).sum()); n += a.shape[0]
full = full.to("cpu"); model.to("cpu")
for cue, exp in es.SPEC["cloze_repo_A"][:10]:
    tid = hl.tok_ids(exp)[0]
    ids = hl._as_row(hl.tok(cue))
    for m, acc in ((model, ranks16), (full, ranks32)):
        acc.append(hl.rank_of(hl.last_logits(m, ids, "cpu"), tid))
print(f"bf16 vs fp32 greedy agreement on {n} held-out windows: {agree/n:.1%}")
print(f"cloze ranks bf16 {ranks16}\n           fp32  {ranks32}")
