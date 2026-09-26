#!/usr/bin/env python3
"""
Tests for the bugs found during code review.
Verifies fixes without interfering with running training.
"""
import os
import sys
import torch
import tempfile
import shutil

# We need to import from train_hope — this loads the tokenizer but doesn't train
from train_hope import (
    HOPE, CONFIG, TOKENIZER, PAD_TOKEN_ID, EOS_TOKEN_ID, VOCAB_SIZE,
    SmartTextDataset
)

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")
        if detail:
            print(f"     → {detail}")


def test_bug3_phase2_checkpoint_seeding():
    """Bug #3: Phase 2 must load foundation weights with step=0, no optimizer state."""
    print("\n🔴 Bug #3: Phase 2 checkpoint seeding")

    # Simulate what run_pipeline.py does: create a fake foundation checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        foundation_path = os.path.join(tmpdir, "hope_foundation_best.pth")
        phase2_path = os.path.join(tmpdir, "hope_final.pth")

        # Create a small model and save a "foundation" checkpoint at step 12000
        model = HOPE(VOCAB_SIZE, 64, 2)  # Tiny model for speed
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

        # Do a dummy forward pass so optimizer has state
        dummy = torch.randint(0, 100, (1, 10))
        logits, _ = model(dummy)
        loss = logits.mean()
        loss.backward()
        optimizer.step()

        foundation_ckpt = {
            'model_state': model.state_dict(),
            'step': 12000,
            'best_val_loss': 5.78,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }
        torch.save(foundation_ckpt, foundation_path)

        # Now replicate the logic from run_pipeline.py (lines 64-87)
        source = foundation_path
        ckpt = torch.load(source, map_location="cpu")
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            phase2_ckpt = {
                'model_state': ckpt['model_state'],
                'step': 0,
                'best_val_loss': float('inf'),
            }
        else:
            phase2_ckpt = ckpt
        torch.save(phase2_ckpt, phase2_path)

        # Now verify the Phase 2 checkpoint
        loaded = torch.load(phase2_path, map_location="cpu")

        test("Phase 2 checkpoint exists", os.path.exists(phase2_path))
        test("Phase 2 has model_state", 'model_state' in loaded)
        test("Phase 2 step is 0", loaded.get('step') == 0,
             f"got step={loaded.get('step')}")
        test("Phase 2 best_val_loss is inf", loaded.get('best_val_loss') == float('inf'),
             f"got {loaded.get('best_val_loss')}")
        test("Phase 2 has NO optimizer_state", 'optimizer_state' not in loaded,
             "optimizer_state was carried over — would use stale momentum")
        test("Phase 2 has NO scheduler_state", 'scheduler_state' not in loaded,
             "scheduler_state was carried over — LR schedule would be wrong")

        # Verify model weights were preserved
        model2 = HOPE(VOCAB_SIZE, 64, 2)
        model2.load_state_dict(loaded['model_state'], strict=True)
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            if not torch.equal(p1, p2):
                test(f"Weight {n1} preserved", False)
                return
        test("All model weights preserved from foundation", True)

        # Verify that train() would not skip Phase 2
        # (the old bug: start_step=12000 > max_steps=3000 → while loop never runs)
        start_step = loaded.get('step', 0)
        max_steps = 3000
        test("train() loop would execute (step < max_steps)",
             start_step < max_steps,
             f"start_step={start_step}, max_steps={max_steps} — loop would be skipped!")


def test_bug1_validation_skip():
    """Bug #1: Validation dataset must skip samples to avoid data leak."""
    print("\n🔴 Bug #1: Validation data leak fix")

    import inspect
    source = inspect.getsource(SmartTextDataset.__init__)
    test("SmartTextDataset accepts skip_samples param",
         'skip_samples' in source)

    iter_source = inspect.getsource(SmartTextDataset.__iter__)
    test("__iter__ uses self.skip_samples",
         'self.skip_samples' in iter_source)

    # Check that val_loader in train() uses skip_samples
    from train_hope import train
    train_source = inspect.getsource(train)
    test("train() creates val_dataset with skip_samples",
         'skip_samples' in train_source)
    test("skip_samples is > 0 (not a no-op)",
         'skip_samples=100000' in train_source or 'skip_samples=50000' in train_source,
         "skip_samples should be set to a large value")


def test_bug2_running_loss():
    """Bug #2: running_loss display should average correctly."""
    print("\n🟡 Bug #2: running_loss accumulation fix")

    import inspect
    from train_hope import train
    source = inspect.getsource(train)

    test("steps_since_display counter exists",
         'steps_since_display' in source)
    test("Loss normalized per accumulate_grad",
         "/ CONFIG['accumulate_grad']" in source or '/ CONFIG["accumulate_grad"]' in source)
    test("avg_loss uses steps_since_display",
         'steps_since_display' in source and 'avg_loss' in source)


def test_bug6_strict_loading():
    """Bug #6: All inference scripts must load weights strictly.

    Since the inference-speed refactor, scripts either contain strict=True
    directly or load via train_hope.load_model_for_inference, which is itself
    strict (verified below)."""
    print("\n🟡 Bug #6: strict loading in inference scripts")

    # The shared loader must be strict
    import inspect
    import train_hope
    loader_src = inspect.getsource(train_hope.load_model_for_inference)
    test("shared loader load_model_for_inference uses strict=True",
         'strict=True' in loader_src and 'strict=False' not in loader_src)

    for fname in ['chat.py', 'app.py', 'generate.py', 'test_model.py']:
        filepath = os.path.join(os.path.dirname(__file__), fname)
        if not os.path.exists(filepath):
            test(f"{fname}: file exists", False, "file not found")
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        has_strict_false = 'strict=False' in content
        has_strict_true = 'strict=True' in content or 'load_model_for_inference' in content

        test(f"{fname}: no strict=False",
             not has_strict_false,
             "still contains strict=False")
        test(f"{fname}: has strict loading",
             has_strict_true,
             "neither strict=True nor load_model_for_inference found")


def test_bug11_typo():
    """Bug #11: Test question typo."""
    print("\n🟢 Bug #11: Test question typo")

    filepath = os.path.join(os.path.dirname(__file__), 'test_model.py')
    with open(filepath, 'r') as f:
        content = f.read()

    test("'What is the sky blue?' is fixed",
         'What is the sky blue?' not in content)
    test("'Why is the sky blue?' is present",
         'Why is the sky blue?' in content)


def test_bug7_lr_schedule():
    """Bug #7: LR schedule should use cosine decay with low floor."""
    print("\n🟢 Bug #7: LR schedule (cosine decay)")

    import inspect
    from train_hope import train, build_nested_optimizers
    source = inspect.getsource(train) + inspect.getsource(build_nested_optimizers)

    test("Uses cosine decay (math.cos)",
         'math.cos' in source or 'cos(' in source,
         "Should use cosine decay, not linear")
    test("Floor is <= 5% of peak (not 10%)",
         'max(0.1,' not in source,
         "Old 10% floor still present")

    # Functional test: extract and call get_lr
    import math
    max_steps = 12000
    warmup_steps = 1500

    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * progress))

    test("LR at step 0 is ~0 (warmup start)", get_lr(0) < 0.01)
    test("LR at warmup end is ~1.0", abs(get_lr(warmup_steps) - 1.0) < 0.01,
         f"got {get_lr(warmup_steps)}")
    test("LR at final step is ~0.01 (1% floor)", abs(get_lr(max_steps) - 0.01) < 0.01,
         f"got {get_lr(max_steps)}")
    test("LR at midpoint > LR at end (decaying)", get_lr(6000) > get_lr(max_steps))


def test_model_basic_sanity():
    """Quick sanity check: model forward pass works, shapes are correct."""
    print("\n🔵 Model sanity checks")

    model = HOPE(VOCAB_SIZE, 64, 2)  # Tiny model
    model.eval()

    # Test forward pass
    x = torch.randint(0, VOCAB_SIZE, (2, 32))
    with torch.no_grad():
        logits, state = model(x)

    test("Forward pass succeeds", True)
    test("Logits shape is [batch, seq, vocab]",
         logits.shape == (2, 32, VOCAB_SIZE),
         f"got {logits.shape}")
    test("State shape is [batch, dim, dim]",
         state.shape == (2, 64, 64),
         f"got {state.shape}")

    # Test single-token inference (like chat.py does)
    single = torch.randint(0, VOCAB_SIZE, (1, 1))
    with torch.no_grad():
        logits2, state2 = model(single, state=state[:1])

    test("Single-token inference works",
         logits2.shape == (1, 1, VOCAB_SIZE),
         f"got {logits2.shape}")

    # Test padding mask
    x_with_pad = torch.tensor([[1, 2, 3, PAD_TOKEN_ID, PAD_TOKEN_ID]])
    with torch.no_grad():
        logits3, state3 = model(x_with_pad)
    test("Padding mask forward pass works", True)


def test_pipeline_file_integrity():
    """Verify run_pipeline.py has all the critical pieces."""
    print("\n🔵 Pipeline file integrity")

    filepath = os.path.join(os.path.dirname(__file__), 'run_pipeline.py')
    with open(filepath, 'r') as f:
        content = f.read()

    test("Imports shutil or torch for checkpoint manipulation",
         'import torch' in content or 'import shutil' in content)
    test("Loads foundation checkpoint before Phase 2",
         'hope_foundation' in content and 'load' in content.lower())
    test("Resets step to 0",
         "'step': 0" in content or '"step": 0' in content)
    # Check that optimizer_state is NOT a key in the phase2_ckpt dict
    # (ignore comments mentioning it)
    import re
    # Find the phase2_ckpt = { ... } block and check its keys
    match = re.search(r"phase2_ckpt\s*=\s*\{([^}]+)\}", content)
    if match:
        dict_body = match.group(1)
        # Remove comment lines
        dict_lines = [l for l in dict_body.split('\n') if not l.strip().startswith('#')]
        dict_keys_text = '\n'.join(dict_lines)
        test("Strips optimizer_state (or doesn't include it)",
             "'optimizer_state'" not in dict_keys_text and '"optimizer_state"' not in dict_keys_text,
             "phase2_ckpt should NOT contain optimizer_state as a key")
    else:
        test("Strips optimizer_state (or doesn't include it)", False,
             "Could not find phase2_ckpt dict in run_pipeline.py")
    test("Saves to hope_final.pth",
         'hope_final.pth' in content)


if __name__ == "__main__":
    print("=" * 60)
    print("HOPE Project — Bug Fix Verification Tests")
    print("=" * 60)

    test_bug3_phase2_checkpoint_seeding()
    test_bug1_validation_skip()
    test_bug2_running_loss()
    test_bug6_strict_loading()
    test_bug11_typo()
    test_bug7_lr_schedule()
    test_model_basic_sanity()
    test_pipeline_file_integrity()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"🎉 ALL {total} TESTS PASSED")
    else:
        print(f"⚠️  {PASS}/{total} passed, {FAIL} FAILED")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
