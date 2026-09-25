#!/usr/bin/env python3
"""
Equivalence tests for the chunk-parallel fast-memory scan.

The chunked path (_chunked_delta_scan) must be MATHEMATICALLY IDENTICAL to the
per-token loop in SelfModifyingLayer._forward_loop — same outputs, same final
memory state, same gradients, including padding masks and sequences whose
length is not divisible by the chunk size.

  1. Direct scan equivalence over many (T, chunk_size, mask, gate) combinations
  2. End-to-end layer equivalence: chunked full-sequence == token-by-token loop
  3. Gradient equivalence: chunked backward == chained single-token backward
  4. Dispatch sanity: T==0 early return, T==1 uses the loop and matches it
"""
import sys
import torch
import torch.nn.functional as F

from train_hope import (
    CONFIG, SelfModifyingLayer, _chunked_delta_scan,
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


def naive_loop_scan(q, k, v, alpha, beta, memory, mask):
    """Reference per-token recurrence (independent re-implementation)."""
    B, T, _ = q.shape
    outs = []
    for t in range(T):
        q_t, k_t, v_t = q[:, t:t+1], k[:, t:t+1], v[:, t:t+1]
        a_t = alpha[:, t].view(B, 1, 1)
        b_t = beta[:, t].view(B, 1, 1)
        outs.append(torch.bmm(q_t, memory))
        err = v_t - torch.bmm(k_t, memory)
        new_memory = a_t * memory + b_t * torch.bmm(k_t.transpose(1, 2), err)
        if mask is not None:
            m_t = mask[:, t].view(B, 1, 1)
            memory = (1 - m_t) * memory + m_t * new_memory
        else:
            memory = new_memory
    return torch.cat(outs, dim=1), memory


def random_case(seed, B=2, D=32, T=17, alpha_bias=0.0, beta_bias=0.0, mask_mode=None):
    torch.manual_seed(seed)
    q = torch.randn(B, T, D)
    k = F.normalize(torch.randn(B, T, D), dim=-1)
    v = torch.randn(B, T, D)
    alpha = torch.sigmoid(torch.randn(B, T) + alpha_bias)
    beta = torch.sigmoid(torch.randn(B, T) + beta_bias)
    M0 = torch.randn(B, D, D) * 0.1
    mask = None
    if mask_mode == "random":
        mask = (torch.rand(B, T) > 0.3).float()
    elif mask_mode == "trailing":
        mask = torch.ones(B, T); mask[:, T // 2:] = 0.0
    elif mask_mode == "zeros":
        mask = torch.zeros(B, T)
    elif mask_mode == "ones":
        mask = torch.ones(B, T)
    return q, k, v, alpha, beta, M0, mask


def test_direct_scan_equivalence():
    print("\n[1] Direct scan equivalence: chunked == per-token loop (float64)")
    worst_out, worst_state = 0.0, 0.0
    cases = 0
    for T in (1, 2, 3, 17, 64, 65, 130):
        for C in (1, 3, 8, 64):
            for mask_mode in (None, "random", "trailing", "zeros", "ones"):
                for a_bias, b_bias in ((0.0, 0.0), (-8.0, 2.0), (8.0, 4.0)):
                    q, k, v, alpha, beta, M0, mask = random_case(
                        seed=T * 1000 + C * 10 + (a_bias * b_bias == 0.0),
                        T=T, alpha_bias=a_bias, beta_bias=b_bias, mask_mode=mask_mode)
                    # float64: the two paths differ only in operation order, so
                    # 64-bit isolates the math (fp32 drift reaches ~1e-4 at T=65
                    # with extreme gates; in fp64 it is ~1e-13).
                    q, k, v, alpha, beta, M0 = [t.double() for t in (q, k, v, alpha, beta, M0)]
                    o_ref, m_ref = naive_loop_scan(q, k, v, alpha, beta, M0.clone(), mask)
                    o_ch, m_ch = _chunked_delta_scan(
                        q, k, v, torch.log(alpha).unsqueeze(-1), beta.unsqueeze(-1),
                        M0.clone(), mask, C)
                    worst_out = max(worst_out, (o_ref - o_ch).abs().max().item())
                    worst_state = max(worst_state, (m_ref - m_ch).abs().max().item())
                    cases += 1
    test(f"{cases} (T, chunk, mask, gate) combinations match (atol 1e-9)",
         worst_out < 1e-9 and worst_state < 1e-9 and
         all(torch.isfinite(m).all() for m in (m_ref, m_ch)),
         f"worst out diff {worst_out:.2e}, worst state diff {worst_state:.2e}")
    print(f"       worst out diff {worst_out:.2e}, worst state diff {worst_state:.2e}")


def test_layer_end_to_end():
    print("\n[2] Layer end-to-end: chunked full-sequence == token-by-token (T=17, C=8)")
    saved = CONFIG.get("fast_chunk_size")
    CONFIG["fast_chunk_size"] = 8
    try:
        torch.manual_seed(0)
        layer = SelfModifyingLayer(32)
        layer.eval()
        x = torch.randn(2, 17, 32)
        mask = (torch.rand(2, 17) > 0.25).float()

        with torch.no_grad():
            out_full, state_full = layer(x, mask=mask)
            state = None
            steps = []
            for t in range(17):
                o_t, state = layer(x[:, t:t+1], state=state, mask=mask[:, t:t+1])
                steps.append(o_t)
            out_inc = torch.cat(steps, dim=1)

        test("logits identical (atol 1e-5)",
             (out_full - out_inc).abs().max().item() < 1e-5,
             f"max diff {(out_full - out_inc).abs().max().item():.2e}")
        test("final memory identical (atol 1e-5)",
             (state_full - state).abs().max().item() < 1e-5,
             f"max diff {(state_full - state).abs().max().item():.2e}")
    finally:
        CONFIG["fast_chunk_size"] = saved


def test_gradient_equivalence():
    print("\n[3] Gradient equivalence: chunked backward == chained single-token backward")
    saved = CONFIG.get("fast_chunk_size")
    CONFIG["fast_chunk_size"] = 8
    try:
        torch.manual_seed(1)
        layer = SelfModifyingLayer(32)
        x = torch.randn(2, 17, 32)

        def run(chunked):
            layer.zero_grad()
            xg = x.detach().requires_grad_(True)
            if chunked:
                out, mem = layer(xg)
            else:
                state, outs = None, []
                for t in range(17):
                    o_t, state = layer(xg[:, t:t+1], state=state)
                    outs.append(o_t)
                out, mem = torch.cat(outs, dim=1), state
            (out.pow(2).mean() + mem.pow(2).mean()).backward()
            return xg.grad.clone(), {n: p.grad.clone() for n, p in layer.named_parameters()}

        gx_chunked, pg_chunked = run(True)
        gx_loop, pg_loop = run(False)

        test("input grad matches (atol 1e-4, rtol 1e-3)",
             torch.allclose(gx_chunked, gx_loop, atol=1e-4, rtol=1e-3),
             f"max diff {(gx_chunked - gx_loop).abs().max().item():.2e}")
        bad = [n for n in pg_loop
               if not torch.allclose(pg_chunked[n], pg_loop[n], atol=1e-4, rtol=1e-3)]
        test("all parameter grads match (atol 1e-4, rtol 1e-3)", not bad, f"mismatched: {bad}")
    finally:
        CONFIG["fast_chunk_size"] = saved


def test_dispatch_sanity():
    print("\n[4] Dispatch sanity")
    saved = dict(chunk=CONFIG.get("fast_chunk_size"), force=CONFIG.get("fast_force_loop"))
    try:
        torch.manual_seed(2)
        layer = SelfModifyingLayer(16)
        layer.eval()

        # T == 0: early return with zeros and untouched memory
        with torch.no_grad():
            m0 = torch.zeros(1, 16, 16)
            out, mem = layer(torch.zeros(1, 0, 16), state=m0)
        test("T==0 returns empty output and unchanged state",
             out.shape == (1, 0, 16) and torch.equal(mem, m0))

        # T == 1: loop path, exact old-formula behavior with carried state
        x1 = torch.randn(1, 1, 16)
        with torch.no_grad():
            out1, mem1 = layer(x1, state=m0.clone())
            q = layer.proj_q(x1); k = F.normalize(layer.proj_k(x1), dim=-1); v = layer.proj_v(x1)
            a = torch.sigmoid(layer.gate_alpha(x1) + 4.0)
            b = torch.sigmoid(layer.gate_beta(x1) - 2.0)
            read = torch.bmm(q, m0)
            mem_ref = a.view(1, 1, 1) * m0 + b.view(1, 1, 1) * torch.bmm(
                k.transpose(1, 2), v - torch.bmm(k, m0))
            out_ref = layer.proj_out(read)
        test("T==1 matches the original single-step formula",
             torch.allclose(out1, out_ref, atol=1e-6) and torch.allclose(mem1, mem_ref, atol=1e-6),
             f"out {(out1 - out_ref).abs().max().item():.2e}, mem {(mem1 - mem_ref).abs().max().item():.2e}")

        # fast_force_loop=True: T>1 falls back to loop and still matches chunked
        xN = torch.randn(2, 9, 16)
        with torch.no_grad():
            CONFIG["fast_force_loop"] = True
            o_loop, m_loop = layer(xN)
            CONFIG["fast_force_loop"] = False
            o_chunk, m_chunk = layer(xN)
        test("fast_force_loop fallback matches chunked path (atol 1e-5)",
             (o_loop - o_chunk).abs().max().item() < 1e-5 and
             (m_loop - m_chunk).abs().max().item() < 1e-5)
    finally:
        CONFIG["fast_chunk_size"] = saved["chunk"]
        CONFIG["fast_force_loop"] = saved["force"]


if __name__ == "__main__":
    print("=" * 60)
    print("HOPE Chunked Scan — Equivalence Tests")
    print("=" * 60)

    test_direct_scan_equivalence()
    test_layer_end_to_end()
    test_gradient_equivalence()
    test_dispatch_sanity()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"{PASS}/{total} passed, {FAIL} FAILED")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
