#!/usr/bin/env python3
"""
Behavioral tests for the nested-learning core:
  1. State-passing equivalence: full-sequence forward == token-by-token forward
     (correctness basis of the O(N) inference claim)
  2. Delta rule: re-presenting a stored (k,v) produces a near-zero second write;
     PAD tokens leave memory bit-identical
  3. Tier schedule: fast tier updates every step, slow tiers only at their period
  4. Checkpoint roundtrip with per-tier optimizer states
"""
import sys
import torch

from train_hope import (
    HOPE, SelfModifyingLayer, CONFIG, PAD_TOKEN_ID, VOCAB_SIZE,
    build_nested_optimizers, make_checkpoint
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


def test_state_passing_equivalence():
    """Full-sequence forward must equal token-by-token state-passing forward."""
    print("\n[1] State-passing equivalence (O(N) inference correctness)")
    torch.manual_seed(0)
    model = HOPE(VOCAB_SIZE, 32, 2, cms_tiers=[[2, 1]])
    model.eval()

    x = torch.randint(0, 1000, (2, 17))

    with torch.no_grad():
        logits_full, state_full = model(x)

        state = None
        logits_steps = []
        for t in range(x.shape[1]):
            lt, state = model(x[:, t:t+1], state=state)
            logits_steps.append(lt)
        logits_inc = torch.cat(logits_steps, dim=1)

    max_diff = (logits_full - logits_inc).abs().max().item()
    test("logits identical (atol 1e-4)", max_diff < 1e-4, f"max diff {max_diff:.2e}")
    state_diff = (state_full - state).abs().max().item()
    test("final memory state identical (atol 1e-4)", state_diff < 1e-4, f"max diff {state_diff:.2e}")


def test_delta_rule():
    """The memory write must be error-driven: storing the same association
    twice produces a much smaller second write."""
    print("\n[2] Delta rule (inner-loop learning, not mere accumulation)")
    torch.manual_seed(0)
    layer = SelfModifyingLayer(32)
    layer.eval()

    x = torch.randn(1, 1, 32)

    # Repeated presentation of the SAME token: the memory's prediction
    # error for (k -> v) must shrink (inner-loop SGD convergence).
    # Plain accumulation (the old M += k^T v) would grow without bound.
    with torch.no_grad():
        k = torch.nn.functional.normalize(layer.proj_k(x), dim=-1)
        v = layer.proj_v(x)
        state = None
        errors = []
        for _ in range(30):
            _, state = layer(x, state=state)
            pred = torch.bmm(k, state)
            errors.append((v - pred).abs().sum().item())

    test("prediction error shrinks over repeats (inner-loop learning)",
         errors[-1] < 0.5 * errors[0],
         f"err[0]={errors[0]:.4f}, err[29]={errors[-1]:.4f}")
    test("memory stays bounded (no blow-up)",
         errors[-1] == errors[-1] and state.abs().max().item() < 100,
         f"max |M| = {state.abs().max().item():.2f}")

    # PAD tokens must leave memory untouched (up to float assoc noise)
    with torch.no_grad():
        pad_x = torch.randn(1, 3, 32)
        mask = torch.tensor([[1.0, 0.0, 0.0]])
        _, m_masked = layer(pad_x, state=None, mask=mask)
        _, m_first_only = layer(pad_x[:, :1, :], state=None, mask=mask[:, :1])
    test("masked tokens leave memory untouched (atol 1e-6)",
         torch.allclose(m_masked, m_first_only, atol=1e-6),
         f"max diff {(m_masked - m_first_only).abs().max().item():.2e}")


def test_tier_schedule():
    """Fast tier params must change every step; slow tier only at its period."""
    print("\n[3] Nested tier update schedule")
    torch.manual_seed(0)

    saved_cfg = {k: CONFIG[k] for k in ('cms_tiers', 'learning_rate', 'max_steps', 'warmup_steps')}
    CONFIG.update({"cms_tiers": [[2, 1], [1, 3]], "learning_rate": 1e-2,
                   "max_steps": 100, "warmup_steps": 0})
    try:
        model = HOPE(VOCAB_SIZE, 32, 3, cms_tiers=CONFIG['cms_tiers'])
        optimizers, schedulers, periods = build_nested_optimizers(model)
        tier_params = [params for _, params in model.tier_param_groups()]
        grad_buffers = [[torch.zeros_like(p) for p in params] for params in tier_params]
        counters = [0] * len(periods)

        test("two tiers built", len(optimizers) == 2)
        test("periods are [1, 3]", periods == [1, 3], f"got {periods}")

        fast_probe = model.cms_layers[0].net[0].weight   # tier 0
        slow_probe = model.cms_layers[2].net[0].weight   # tier 1

        slow_change_steps = []
        fast_change_steps = []
        for step in range(1, 7):
            fast_before = fast_probe.detach().clone()
            slow_before = slow_probe.detach().clone()

            model.zero_grad(set_to_none=True)
            x = torch.randint(0, 1000, (2, 8))
            logits, _ = model(x)
            loss = logits.float().pow(2).mean()
            loss.backward()

            for i, (opt, sch, period, params) in enumerate(
                    zip(optimizers, schedulers, periods, tier_params)):
                for buf, p in zip(grad_buffers[i], params):
                    if p.grad is not None:
                        buf.add_(p.grad)
                counters[i] += 1
                if counters[i] >= period:
                    for buf, p in zip(grad_buffers[i], params):
                        p.grad = buf.div_(period)
                    opt.step()
                    for j in range(len(grad_buffers[i])):
                        grad_buffers[i][j] = torch.zeros_like(params[j])
                    counters[i] = 0
                sch.step()

            if not torch.equal(fast_before, fast_probe):
                fast_change_steps.append(step)
            if not torch.equal(slow_before, slow_probe):
                slow_change_steps.append(step)

        test("fast tier updated every step (1-6)",
             fast_change_steps == [1, 2, 3, 4, 5, 6], f"got {fast_change_steps}")
        test("slow tier updated only at steps 3 and 6",
             slow_change_steps == [3, 6], f"got {slow_change_steps}")
    finally:
        CONFIG.update(saved_cfg)


def test_checkpoint_roundtrip():
    """make_checkpoint must carry per-tier optimizer/scheduler states."""
    print("\n[4] Checkpoint roundtrip (multi-optimizer)")
    import tempfile, os
    torch.manual_seed(0)

    saved_cfg = {k: CONFIG[k] for k in ('cms_tiers',)}
    CONFIG.update({"cms_tiers": [[1, 1], [1, 4]]})
    try:
        model = HOPE(VOCAB_SIZE, 32, 2, cms_tiers=CONFIG['cms_tiers'])
        optimizers, schedulers, periods = build_nested_optimizers(model)

        ckpt = make_checkpoint(model, optimizers, schedulers, [0, 2], 42, 3.14)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pth")
            torch.save(ckpt, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)

        test("has optimizer_states list (one per tier)",
             isinstance(loaded.get('optimizer_states'), list) and len(loaded['optimizer_states']) == 2)
        test("has scheduler_states list", len(loaded.get('scheduler_states', [])) == 2)
        test("has tier_counters", loaded.get('tier_counters') == [0, 2])
        test("records cms_tiers config", loaded.get('cms_tiers') == [[1, 1], [1, 4]])
        test("step preserved", loaded.get('step') == 42)

        model2 = HOPE(VOCAB_SIZE, 32, 2, cms_tiers=CONFIG['cms_tiers'])
        model2.load_state_dict(loaded['model_state'], strict=True)
        test("model_state loads strict=True", True)
    finally:
        CONFIG.update(saved_cfg)


if __name__ == "__main__":
    print("=" * 60)
    print("HOPE Nested Learning — Behavioral Tests")
    print("=" * 60)

    test_state_passing_equivalence()
    test_delta_rule()
    test_tier_schedule()
    test_checkpoint_roundtrip()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"{PASS}/{total} passed, {FAIL} FAILED")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
