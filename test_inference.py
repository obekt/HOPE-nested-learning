#!/usr/bin/env python3
"""
Tests for the inference optimizations:
  1. last_only prefill equivalence: final-position logits identical to full forward
     (CMS blocks are position-wise, so slicing before them must not change the
     last position's output), including with carried state
  2. Sampler: shape, temperature-0-ish greedy agreement, repetition penalty
     suppresses, top-p/top-k produce in-vocab ids, no NaN (MPS clamp path)
  3. bf16 model: finite outputs, close to fp32 on CPU small model
"""
import sys
import torch

from train_hope import (
    HOPE, VOCAB_SIZE, PAD_TOKEN_ID, sample_next_token,
)

PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")
        if detail:
            print(f"       -> {detail}")


def test_last_only_equivalence():
    print("\n[1] last_only prefill == full forward at the final position")
    torch.manual_seed(0)
    model = HOPE(VOCAB_SIZE, 32, 2, cms_tiers=[[2, 1]])
    model.eval()
    x = torch.randint(0, 1000, (2, 17))

    with torch.no_grad():
        logits_full, state_full = model(x)
        logits_last, state_last = model(x, last_only=True)
    test("last_only logits shape [B, 1, V]", logits_last.shape == (2, 1, VOCAB_SIZE),
         f"got {tuple(logits_last.shape)}")
    diff = (logits_full[:, -1, :] - logits_last[:, 0, :]).abs().max().item()
    test("final-position logits identical (atol 1e-4)", diff < 1e-4, f"max diff {diff:.2e}")
    sdiff = (state_full - state_last).abs().max().item()
    test("memory state identical (atol 1e-4)", sdiff < 1e-4, f"max diff {sdiff:.2e}")

    # with carried state (second prefill chunk of a longer context)
    with torch.no_grad():
        x2 = torch.randint(0, 1000, (2, 5))
        lf, sf = model(x2, state=state_full)
        ll, sl = model(x2, state=state_full, last_only=True)
    diff2 = (lf[:, -1, :] - ll[:, 0, :]).abs().max().item()
    test("equivalence holds with carried state (atol 1e-4)",
         diff2 < 1e-4 and (sf - sl).abs().max().item() < 1e-4, f"max diff {diff2:.2e}")


def test_sampler():
    print("\n[2] Sampler behavior")
    torch.manual_seed(0)
    V = VOCAB_SIZE
    logits = torch.randn(1, 1, V)

    tok = sample_next_token(logits, temperature=0.7, top_k=40, top_p=0.9)
    test("returns [B,1] in-vocab token", tok.shape == (1, 1) and 0 <= tok.item() < V)

    # near-greedy: very low temperature should pick the argmax
    greedy = sample_next_token(logits, temperature=0.01, top_k=0, top_p=1.0)
    test("temperature~0 picks argmax", greedy.item() == logits[0, -1].argmax().item())

    # repetition penalty: a slightly-weaker fresh competitor must win once the
    # dominant token is penalized (deterministic at near-greedy temperature)
    comp = torch.full((1, 1, V), -20.0)
    comp[0, 0, 777] = 3.0   # dominant, but in prev_tokens
    comp[0, 0, 888] = 2.9   # fresh competitor
    no_pen = sample_next_token(comp, temperature=0.01, top_k=2, top_p=1.0,
                               repetition_penalty=1.0)
    test("without penalty the dominant token wins", no_pen.item() == 777,
         f"got {no_pen.item()}")
    with_pen = sample_next_token(comp, temperature=0.01, top_k=2, top_p=1.0,
                                 repetition_penalty=1.3, prev_tokens=[777])
    test("repetition penalty flips to the fresh competitor (3.0/1.3 < 2.9)",
         with_pen.item() == 888, f"got {with_pen.item()}")
    before = comp.clone()
    sample_next_token(comp, temperature=0.7, top_k=40, top_p=0.9,
                      repetition_penalty=1.2, prev_tokens=[777, 888])
    test("sampler does not mutate caller logits", torch.equal(before, comp))

    # degenerate distributions must not produce NaN (MPS multinomial guard)
    flat = torch.zeros(1, 1, V)
    tok_flat = sample_next_token(flat, temperature=1.0, top_k=40, top_p=0.9)
    test("flat logits: finite sample, no NaN", 0 <= tok_flat.item() < V)
    neg_inf = torch.full((1, 1, V), float('-inf')); neg_inf[0, 0, 5] = 0.0
    tok_ni = sample_next_token(neg_inf, temperature=1.0, top_k=10, top_p=0.5)
    test("all -inf except one: picks the survivor", tok_ni.item() == 5)


def test_bf16_model():
    print("\n[3] bf16 inference numerics (CPU small model)")
    torch.manual_seed(1)
    model = HOPE(VOCAB_SIZE, 32, 2, cms_tiers=[[2, 1]])
    model.eval()
    x = torch.randint(0, 1000, (1, 9))

    with torch.no_grad():
        logits32, state32 = model(x)
        model_bf = HOPE(VOCAB_SIZE, 32, 2, cms_tiers=[[2, 1]])
        model_bf.load_state_dict(model.state_dict())
        model_bf.to(torch.bfloat16).eval()
        xb = x
        logits_bf, state_bf = model_bf(xb)

    test("bf16 outputs finite", torch.isfinite(logits_bf).all() and torch.isfinite(state_bf).all())
    test("bf16 model keeps fp32 memory state (MPS solve_triangular is fp32-only)",
         state_bf.dtype == torch.float32, f"got {state_bf.dtype}")
    ld = (logits32[:, -1, :] - logits_bf[:, -1, :].float()).abs().max().item()
    sd = (state32 - state_bf.float()).abs().max().item()
    scale = logits32.abs().max().item()
    test(f"bf16 close to fp32 (logit max diff {ld:.3f} vs scale {scale:.1f})",
         ld < 0.15 * max(scale, 1.0) and sd < 0.5, f"logits {ld:.3f}, state {sd:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("HOPE Inference Optimizations — Tests")
    print("=" * 60)

    test_last_only_equivalence()
    test_sampler()
    test_bf16_model()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"{PASS}/{total} passed, {FAIL} FAILED")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
