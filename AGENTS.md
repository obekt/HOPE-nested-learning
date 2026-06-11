# AGENTS.md — HOPE Nested Learning

This file exists so any AI agent (or human developer) can quickly understand, modify, and improve this project.

---

## 🎯 Project Purpose

Implement the **HOPE (Hierarchical Optimization with Progressive Encoding)** architecture from the paper *"Nested Learning: The Illusion of Deep Learning"* (Behrouz et al., 2024).

Core idea: Intelligence is a nested optimization problem, not just deep layers. The model has:
- **Fast Weights** (Self-Modifying Layer): Adapts in real-time to input context
- **Slow Weights** (Continuum Memory System / CMS): Stores long-term knowledge

---

## 📁 File Structure

| File | Purpose |
|------|---------|
| `train_hope.py` | **Core file.** Model architecture, dataset loader, training loop, dashboard. All other scripts import from here. |
| `run_pipeline.py` | Automated full pipeline: Wikipedia foundation → Q&A fine-tuning. Runs both phases back-to-back. |
| `chat.py` | Interactive console chat with the model. Supports `/temp`, `/tokens`, `/reasoning` commands. |
| `app.py` | Gradio web UI for chatting with the model. |
| `generate.py` | CLI one-shot text generation. |
| `test_model.py` | Quick evaluation on 5 hardcoded test questions. |
| `test_nested.py` | **Behavioral tests** for the nested-learning core: state-passing equivalence, delta-rule convergence, tier update schedule, checkpoint roundtrip. Run after any change to the model or training loop. |
| `test_fixes.py` | Regression tests from earlier code reviews. |
| `requirements.txt` | Python dependencies. |

---

## 🏗️ Architecture Details

### Model: `HOPE` class

```
Input Tokens
    ↓
Embedding (vocab_size × d_model)
    ↓
SelfModifyingLayer (Fast Weights — inner-loop delta rule)
    - proj_q, proj_k, proj_v, proj_out (Linear layers); keys L2-normalized
    - gate_alpha (per-token forget gate), gate_beta (per-token inner LR)
    - Memory update: M_t = α_t·M_{t-1} + β_t·k_tᵀ(v_t − k_t·M_{t-1})
      → one SGD step per token on ||kM − v||²; writes only prediction error
    - Maintains memory matrix: [batch, d_model, d_model]
    - Optional padding mask: masked tokens leave M untouched
    ↓
LayerNorm + Residual
    ↓
CMS Layers × n_layers (Slow Weights, tiered update frequencies)
    - Each: Linear → GELU → Linear → Dropout, LayerNorm + Residual
    - Partitioned by CONFIG["cms_tiers"] = [[8,1],[5,4],[3,16]]:
      tier 0 (8 layers + embedding/fast_memory/head) steps every opt step,
      tier 1 (5 layers) every 4 steps, tier 2 (3 layers) every 16 steps.
    - Slow tiers apply the AVERAGE of buffered gradients when they step.
    ↓
Head (d_model × vocab_size)
    ↓
Logits
```

### Key Design Decisions

1. **Byte-level → GPT-2 Tokenizer**: We switched from 256-byte vocab to GPT-2's 50K BPE tokenizer. This made training 10-50x more efficient. The architecture itself is unchanged.
2. **Padding Mask**: `SelfModifyingLayer` takes a `mask` parameter. When `isolate_samples=True`, real tokens are 1 and padding is 0. Memory only updates on real tokens.
3. **State Passing**: During inference, the memory matrix from the SelfModifyingLayer is carried forward token-by-token. This gives O(N) inference instead of O(N²). Verified by an equivalence test in `test_nested.py`.
4. **Nested optimizers**: `build_nested_optimizers()` creates one AdamW + LR scheduler per tier. The training loop buffers gradients per tier and steps each tier at its period (`train_hope.py`, "NESTED UPDATE" block). Checkpoints store `optimizer_states` (list), `scheduler_states` (list), and `tier_counters`. Old single-optimizer checkpoints still load (model weights only, fresh optimizers).
5. **Delta-rule memory**: error-driven writes mean repeated content converges instead of accumulating; memory stays bounded without a normalizer.

---

## ⚙️ Configuration

All training config lives in the `CONFIG` dict at the top of `train_hope.py`.

**Critical fields:**

```python
CONFIG = {
    "d_model": 512,           # Width. More = smarter but slower
    "n_layers": 16,           # Depth. More = deeper reasoning
    "vocab_size": 50257,      # GPT-2 tokenizer vocab size. DO NOT CHANGE.
    "seq_len": 512,           # Context window
    "batch_size": 4,
    "accumulate_grad": 8,     # Effective batch = batch_size × accumulate_grad
    "learning_rate": 2e-4,    # Foundation rate
    "max_steps": 12000,       # Total optimizer steps
    "warmup_steps": 1500,     # Linear warmup
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    "isolate_samples": False, # True = each row is separate (Q&A). False = packed (Wikipedia)
    "save_path": "hope_foundation.pth",
    "checkpoint_every": 1000,
    "cms_tiers": [[8, 1], [5, 4], [3, 16]],  # Nested tiers: [n_layers, update_period]. Counts must sum to n_layers.
}
```

### Tokenizer Globals

At module level in `train_hope.py`:
```python
TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
TOKENIZER.pad_token = TOKENIZER.eos_token
PAD_TOKEN_ID = 50256
EOS_TOKEN_ID = 50256
VOCAB_SIZE = 50257
```

**All scripts import these from `train_hope.py`.**

---

## 🚀 How to Train

### Option A: Full Pipeline (Recommended)

```bash
python run_pipeline.py
```

This runs:
1. Wikipedia foundation (12K steps, `hope_foundation.pth`)
2. Auto-switches to Q&A fine-tuning (3K steps, `hope_final.pth`)

**To monitor:**
```bash
tail -f foundation.log    # Phase 1
tail -f finetune.log      # Phase 2
tail -f pipeline.log      # Combined stdout
```

### Option B: Manual Training

Edit `CONFIG` in `train_hope.py`, then:
```bash
python train_hope.py
```

The script auto-resumes from `save_path` if it exists.

---

## 🧪 How to Test / Inference

After training produces a `.pth` file:

```bash
# Quick test
python test_model.py

# Interactive chat
python chat.py

# Web UI
python app.py

# CLI generation
python generate.py --prompt "Why is the sky blue?" --temperature 0.7
```

**All inference scripts load from `CONFIG['save_path']`** and fall back to `*_best.pth` if the main file doesn't exist.

---

## 🐛 Known Issues & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| `probability tensor contains inf, nan` | MPS numerical instability + top-p sampling | Simplified sampling in `generate.py`. If it recurs, add `torch.clamp(probs, min=1e-10)` before `multinomial`. |
| GPT2Tokenizer max_length warning | Long Wikipedia articles exceed tokenizer's default `model_max_length` | Set `TOKENIZER.model_max_length = 1_000_000_000` |
| Loss flat at ~10.8 for first 500 steps | Random guessing across 50K vocab is hard | Normal. Wait until LR warms up. Should drop by step 1000. |
| Out of Memory on MPS | 86M params + large activations | Reduce `batch_size` to 2 or `seq_len` to 256. |
| Output is gibberish | Model trained without foundation (Q&A only) | Must run full pipeline: Wikipedia first, then Q&A. |

---

## 🔧 How to Improve This Model

### Quick Wins

1. **Train longer** — Increase `max_steps` in `run_pipeline.py`. Foundation benefits greatly from 20K+ steps.
2. **Bigger model** — `d_model=768, n_layers=32` = ~300M params. Much more capable. Requires 32GB+ RAM.
3. **Mixed dataset** — Interleave Wikipedia + Q&A during foundation so the model learns format earlier.

### Medium Effort

4. **Add rotary positional embeddings (RoPE)** — Currently the model has NO positional encoding. Adding RoPE would improve sequence understanding.
5. **Add attention to CMS blocks** — Currently CMS is just FFN. Adding lightweight attention would help.
6. **Better tokenizer** — Train a custom BPE tokenizer on the Q&A dataset for more efficient vocabulary.

### Hard / Research

7. **Weight transfer from GPT-2** — Not directly possible due to architecture mismatch. But could initialize embeddings from GPT-2's embedding layer (same vocab, same size if d_model=768).
8. **Knowledge Distillation** — Use GPT-2 or GPT-4o-mini to generate "teacher" answers for the Q&A dataset. Train HOPE to match teacher logits.
9. ~~**Multi-scale CMS**~~ — **DONE.** Fast/Medium/Slow tiers with per-tier optimizers and update periods (`cms_tiers` in CONFIG, `build_nested_optimizers()`, the NESTED UPDATE block in `train()`). Behavioral tests in `test_nested.py`.
10. **Benchmark forgetting** — Measure Phase 1 val loss before/after Phase 2 to quantify how much the tiered schedule actually mitigates catastrophic forgetting (claim is currently structural, not measured).
11. **Parallelize the fast-memory scan** — The per-token Python loop is the training bottleneck; a chunked/blocked scan would speed it up substantially.
12. **TBPTT across batches** — Memory state is reset every batch (`state=None`); carrying it across windows would train long-horizon memory behavior.

---

## 📝 Git Workflow

```bash
# Before making changes
git pull origin main

# After changes
git add -A
git commit -m "type: description"
```

**Do NOT push unless explicitly asked.**

---

## 🎓 Paper Reference

> **Nested Learning: The Illusion of Deep Learning**  
> Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research)  
> [Paper Link](https://abehrouz.github.io/files/NL.pdf)

This is an **unofficial implementation** for experimentation.

---

## 💡 Agent Notes

- **Always import from `train_hope.py`** — it contains the model, tokenizer, config, and device selection.
- **Never hardcode byte-level encoding** — use `TOKENIZER.encode()` and `TOKENIZER.decode()`.
- **Padding token is EOS token** — `PAD_TOKEN_ID = EOS_TOKEN_ID = 50256`.
- **Loss masking** — Always use `ignore_index=PAD_TOKEN_ID` in cross-entropy.
- **Memory efficiency** — The SelfModifyingLayer stores state per timestep. Long `seq_len` + large `batch_size` = high memory.
- **MPS quirks** — Mac Metal backend sometimes produces NaN in softmax. Use clamping or fallback to CPU if unstable.
