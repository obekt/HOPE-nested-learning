# HOPE: The Nested Learning Experiment 🧠

> **"Deep Learning is an illusion. Real learning is a set of nested optimization problems."**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📖 What is this?

This is a clean, from-scratch PyTorch implementation of the **HOPE architecture**, based on the groundbreaking paper *"Nested Learning: The Illusion of Deep Learning"* (Behrouz et al., 2024).

Standard Large Language Models (LLMs) suffer from **"Anterograde Amnesia"**—once trained, they are frozen. They can't learn from new conversations without a full re-training.

**HOPE changes the paradigm.** Instead of just stacking static layers, it models intelligence as a **Continuum Memory System**:

* **Fast Weights (Self-Modifying Layer):** An inner-loop learner. Its memory matrix performs one step of online gradient descent (a **delta rule**) per token: it writes only the *prediction error* `v − kM`, with a learned per-token forget gate (α) and inner learning rate (β). Content the memory already knows is not re-written.
* **Slow Weights (Continuum Memory):** A stack of FFN blocks partitioned into tiers that are updated by the outer optimizer at **different frequencies** (see below), so later tiers consolidate slowly.

## 🚀 Key Features

* **🧠 Inner-Loop Fast Memory:** Delta-rule fast-weight memory (error-driven writes, learned gates) that adapts to the immediate prompt token-by-token.
* **⚡ Fast State-Passing Inference:** Optimized $O(N)$ generation algorithm that carries model memory forward. Verified by an equivalence test (`test_nested.py`): full-sequence forward ≡ token-by-token forward.
* **🕰️ Continuum Memory System (CMS):** Layers are partitioned into tiers with different optimizer update periods — by default Fast (every step), Medium (every 4 steps), Slow (every 16 steps), with gradients averaged in between. Configured via `cms_tiers` in `CONFIG`.
* **⚡ Ultra-Lightweight:** Designed to run on **Consumer Hardware** (Mac M1/M2/M3/M4, NVIDIA RTX 3060+, or even CPU).
* **🔄 Continual Learning (experimental):** The multi-frequency tiers mean fine-tuning predominantly moves fast tiers while slow tiers consolidate. This mitigates forgetting structurally; it is not yet benchmarked.
* **📱 Consumer Device Ready:** Optimized <1GB RAM footprint for inference on everyday hardware.
* **🛡️ Padding Masking & Memory Integrity:** Binary masking in the self-modifying layers prevents "Padding Leakage," ensuring the model's memory stays pure during fine-tuning on isolated datasets.
* **📝 Automatic Instruction Tuning:** Built-in formatting that turns raw multi-column datasets into structured assistant prompts (Question/Answer/Reasoning).
* **🔤 GPT-2 Tokenizer:** Uses a real subword tokenizer (50K vocab) instead of byte-level, making training 10-50x more efficient.

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
