# HOPE: The Nested Learning Experiment 🧠

> **"Deep Learning is an illusion. Real learning is a set of nested optimization problems."**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

📊 **New: [The HOPE Marathon](EXPERIMENT.md)** — a fully documented 15-hour training experiment on one Mac (48k steps, ~786M tokens): the 5× speedup methodology, hour-by-hour learning curves, factual-knowledge emergence ranks, and an honest negative result on factual recall at consumer data budgets. Trained checkpoints are on the [`v0.1-marathon` release](https://github.com/obekt/HOPE-nested-learning/releases).

## 📖 What is this?

This is a clean, from-scratch PyTorch implementation of the **HOPE architecture**, based on the groundbreaking paper *"Nested Learning: The Illusion of Deep Learning"* (Behrouz et al., 2024).

Standard Large Language Models (LLMs) suffer from **"Anterograde Amnesia"**—once trained, they are frozen. They can't learn from new conversations without a full re-training.

**HOPE changes the paradigm.** Instead of just stacking static layers, it models intelligence as a **Continuum Memory System**:

* **Fast Weights (Self-Modifying Layer):** An inner-loop learner. Its memory matrix performs one step of online gradient descent (a **delta rule**) per token: it writes only the *prediction error* `v − kM`, with a learned per-token forget gate (α) and inner learning rate (β). Content the memory already knows is not re-written.
* **Slow Weights (Continuum Memory):** A stack of FFN blocks partitioned into tiers that are updated by the outer optimizer at **different frequencies** (see below), so later tiers consolidate slowly.

## 🚀 Key Features

* **🧠 Inner-Loop Fast Memory:** Delta-rule fast-weight memory (error-driven writes, learned gates) that adapts to the immediate prompt token-by-token.
* **⚡ Fast State-Passing Inference:** Optimized $O(N)$ generation algorithm that carries model memory forward. Verified by an equivalence test (`test_nested.py`): full-sequence forward ≡ token-by-token forward. On Apple MPS: **bf16 weights** (greedy-identical for *trained* weights; fp32 memory state), **last-position-only prefill** (2.5× faster at T=512), optional `torch.compile` for generation steps, and nucleus+repetition-penalty sampling — ~500 tok/s generation end-to-end.
* **🧮 Chunk-Parallel Fast-Memory Scan:** The delta-rule recurrence (a *gated delta rule*, as in DeltaNet/Gated DeltaNet) is evaluated in an exact chunk-parallel form during training and prompt prefill — 22× faster layer fwd+bwd on Apple MPS (~5× per training step), mathematically identical to the token loop (verified in float64 to ~1e-13 in `test_chunked.py`).
* **🕰️ Continuum Memory System (CMS):** Layers are partitioned into tiers with different optimizer update periods — by default Fast (every step), Medium (every 4 steps), Slow (every 16 steps), with gradients averaged in between. Configured via `cms_tiers` in `CONFIG`.
* **⚡ Ultra-Lightweight:** Designed to run on **Consumer Hardware** (Mac M1/M2/M3/M4, NVIDIA RTX 3060+, or even CPU).
* **🔄 Continual Learning (experimental, and *not* working at this budget):** The multi-frequency tiers are meant to let fine-tuning move fast tiers while slow tiers consolidate. **Measured 2026-11-05: it did not protect anything here** — going from the 48k-step foundation to the 6k-step Q&A fine-tune moved held-out Wikipedia perplexity **421 → 2 552 (6.1× worse)** and in-context copying **11.3% → 0.7%** ([`surgery/README.md`](surgery/README.md), D7). The mechanism is implemented and tested; its *benefit* is unproven and currently contradicted.
* **📱 Consumer Device Ready:** Optimized <1GB RAM footprint for inference on everyday hardware.
* **🛡️ Padding Masking & Memory Integrity:** Binary masking in the self-modifying layers prevents "Padding Leakage," ensuring the model's memory stays pure during fine-tuning on isolated datasets.
* **📝 Automatic Instruction Tuning:** Built-in formatting that turns raw multi-column datasets into structured assistant prompts (Question/Answer/Reasoning).
* **🔤 GPT-2 Tokenizer:** Uses a real subword tokenizer (50K vocab, Rust `GPT2TokenizerFast`) instead of byte-level, making training 10-50x more efficient. A background prefetch thread overlaps streaming + tokenization with GPU compute.

---

## 📏 What the released model can and cannot actually do (measured 2026-11-05)

Numbers below come from a frozen, pre-registered suite (`surgery/src/eval_spec.json`)
run on the released `v0.1-marathon` checkpoints, with an untrained HOPE and GPT-2
small (same 50,257-token vocabulary) put through the **same code** as floor and
reference. Deterministic only — likelihood and greedy, no sampling luck.

| | held-out wiki ppl | 38-cue factual cloze @1/@5 | 25 **fresh** facts @5 | in-context copy | ARC / MMLU |
|---|---|---|---|---|---|
| untrained HOPE | 50 972 | 0.000 / 0.000 | 0.000 | 0.000 | 0.260 / 0.241 |
| `hope_foundation_best` | 421 | 0.026 / 0.079 | **0.000** | 0.113 (+0.010 from context) | 0.202 / 0.256 |
| `hope_final_best` | 2 552 | 0.079 / 0.105 | **0.000** | 0.007 | 0.209 / 0.263 |
| GPT-2 small (reference) | 49 | 0.289 / 0.579 | 0.280 | 0.350 (+0.113) | 0.238 / 0.269 |

**It genuinely does:** write fluent English in Wikipedia's genre; hold a chat format
and emit Answer/Reasoning structure; keep a running per-token memory that is
load-bearing (ablate it and perplexity goes 421 → 1 535); run at ~200–500 tok/s on a
laptop.

**It does not do:** answer questions it was not memorised on (25 fresh facts: 0/25
top-5); reason — multiple-choice likelihood is at chance **for GPT-2 small too** under
this protocol, so those columns say little at 86–124M; or reliably use what is in its
context window (copying 11.3% vs GPT-2's 35%).

**Why (this is the useful part):** auditing the weights shows `head.weight` is the
only well-trained matrix (rms 18.8× its init), while **61% of embedding rows and CMS
blocks 8–15 — the slow tiers — are still at random initialisation**. And ridge probes
of the head's own input recover the current token at R² 0.46 but the **integrated
context at only R² 0.066**: the 16 position-wise CMS blocks destroy most of what the
fast memory integrated before the linear head can read it. That is an architectural
wiring problem, not a training-budget one — see open items 13–14 in
[`AGENTS.md`](AGENTS.md).

**Training-free weight surgery (`surgery/`, D7)** re-solves just the output layer in
closed form and halves held-out perplexity (**421 → 216**; the Q&A checkpoint
**2 552 → 548**) while leaving factual ranking and retrieval where they were — i.e.
language modelling improves, capability does not. It also maps exactly where the
ceiling is, which is why no other edit family (memory geometry, embedding repair,
layer neutralisation, checkpoint merging) survived measurement.

---

## 🛠️ Requirements

You don't need a massive server. This implementation is optimized for **Laptops** and **Home PCs**.

* **Python:** 3.9 or newer
* **Memory:** 8GB RAM minimum (16GB recommended, 32GB+ for larger configs)
* **GPU:** Optional but recommended (NVIDIA CUDA or Mac MPS supported)

### Python Libraries

```bash
pip install -r requirements.txt
```

---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/obekt/HOPE-nested-learning.git
   cd HOPE-nested-learning
   ```

2. **Setup Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚦 Usage

### 1. Full Automated Pipeline (Recommended)

Run foundation training on Wikipedia, then automatic fine-tuning on Q&A:

```bash
python run_pipeline.py
```

*   **Phase 1:** Wikipedia foundation (`hope_foundation.pth`) — 12K steps
*   **Phase 2:** Q&A fine-tuning (`hope_final.pth`) — 3K steps
*   **Output:** Final model saved as `hope_final.pth`
*   **Auto-resume:** If interrupted, each phase resumes from its latest checkpoint

### 2. Train Individual Phases

**Foundation only:**
```bash
python train_hope.py
# Edit CONFIG in train_hope.py for dataset/settings
```

**Fine-tune from foundation:**
```python
# In train_hope.py, set:
# save_path = "hope_foundation.pth"  (to load existing)
# Then switch dataset to Q&A and run again
```

### 3. Quick Test

Evaluate your model on sample questions:
```bash
python test_model.py
```

### 4. Chat in the Console

```bash
python chat.py
```

*   Uses **Fast State-Passing** for $O(N)$ inference.
*   Commands: `/temp 0.7`, `/tokens 250`, `/reasoning off`
*   Type `quit` to exit.

### 5. Run the Web Interface

```bash
python app.py
```

*   **Features:** Temperature slider, max tokens, reasoning toggle.

### 6. Generate Non-Interactively

```bash
python generate.py --prompt "Why is the sky blue?" --temperature 0.7 --max-tokens 200
```

---

## 🧪 Configuration

Edit `CONFIG` in `train_hope.py`:

```python
CONFIG = {
    "d_model": 512,           # Width (Reasoning capability)
    "n_layers": 16,           # Depth
    "seq_len": 512,           # Training window
    "vocab_size": 50257,      # GPT-2 tokenizer
    "batch_size": 4,
    "accumulate_grad": 8,     # Effective batch = 32
    "max_steps": 12000,       # Training steps
    "learning_rate": 2e-4,
    "isolate_samples": False, # True for Q&A, False for Wikipedia

    # Nested learning tiers: [n_layers, update_period] pairs.
    # 8 fast layers (step every opt step), 5 medium (every 4), 3 slow (every 16).
    # Layer counts must sum to n_layers.
    "cms_tiers": [[8, 1], [5, 4], [3, 16]],
}
```

### Preset Configurations:

| Config | Params | Notes |
|--------|--------|-------|
| **Nano** | ~25M | `d_model=384, n_layers=8` — Fastest, for testing |
| **Balanced** | ~86M | `d_model=512, n_layers=16` — **Default**, good for Q&A |
| **Deep** | ~150M | `d_model=512, n_layers=32` — Higher quality, slower |
| **Ultra** | ~300M | `d_model=768, n_layers=32` — High-end hardware only |

---

## 🧪 Training Pipeline

### Phase 1: Foundation (Grammar & Facts)

*   **Dataset:** `wikimedia/wikipedia` (English)
*   **Method:** Packed Training (stitching articles together)
*   **Goal:** Learn English grammar, vocabulary, and general world knowledge
*   **Duration:** ~10-12 hours on M4 Max (12K steps)

### Phase 2: Fine-Tuning (Q&A Format)

*   **Dataset:** `obekt/obekt-question-answer-reasoning-micro-v0.1` (~77K Q&A pairs)
*   **Method:** Sample-Isolated Mode with Padding Masking
*   **Goal:** Learn Question → Answer → Reasoning format
*   **Duration:** ~2-3 hours on M4 Max (3K steps)

### Expected Results

With proper training, the model should:
1. Understand the Q&A format
2. Generate coherent English sentences
3. Optionally produce reasoning explanations
4. Be usable via `chat.py`, `app.py`, or `generate.py`

It should **not** be expected to answer unseen questions: format is learned, factual
generalisation is not (measured 0/25 on fresh held-out facts — see "What the released
model can and cannot actually do" above).

---

## 🧠 Performance & RAM Specs

*   **Training:** ~10-15GB RAM for 86M param model
*   **Inference:** < 2GB RAM (state-passing optimization)

---

## 📜 Credits & Citation

This code is an unofficial implementation and experimental exploration of the concepts introduced in:

> **Nested Learning: The Illusion of Deep Learning**  
> Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research)  
> [Paper Link](https://abehrouz.github.io/files/NL.pdf)
