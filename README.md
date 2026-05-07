# HOPE: The Nested Learning Experiment 🧠

> **"Deep Learning is an illusion. Real learning is a set of nested optimization problems."**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📖 What is this?
This is a clean, from-scratch PyTorch implementation of the **HOPE architecture**, based on the groundbreaking paper *"Nested Learning: The Illusion of Deep Learning"* (Behrouz et al., 2024).

Standard Large Language Models (LLMs) suffer from **"Anterograde Amnesia"**—once trained, they are frozen. They can't learn from new conversations without a full re-training.

**HOPE changes the paradigm.** Instead of just stacking static layers, it models intelligence as a **Continuum Memory System**:
* **Fast Weights (Self-Modifying Layer):** A layer that *updates its own parameters* in real-time as it reads text. It learns your specific context instantly.
* **Slow Weights (Continuum Memory):** Deep layers that update rarely, storing long-term knowledge (grammar, facts) without catastrophic forgetting.

## 🚀 Key Features
* **🧠 Self-Modifying Architecture:** Uses a "Fast Weight" mechanism (Linear Attention dual form) to adapt to the immediate prompt dynamically.
* **⚡ Fast State-Passing Inference:** Optimized $O(N)$ generation algorithm that carries model memory forward, enabling lightning-fast responses even for long sequences.
* **🕰️ Continuum Memory System (CMS):** A hierarchy of layers that update at different frequencies (Fast, Medium, Slow), mimicking the human brain's memory consolidation.
* **⚡ Ultra-Lightweight:** Designed to run on **Consumer Hardware** (Mac M1/M2/M3/M4, NVIDIA RTX 3060+, or even CPU).
* **🔄 Continual Learning:** Capable of training on Dataset A, then Dataset B, without instantly forgetting Dataset A.
* **📱 Consumer Device Ready:** Optimized <1GB RAM footprint for inference on everyday hardware.
* **🛡️ Padding Masking & Memory Integrity:** Binary masking in the self-modifying layers prevents "Padding Leakage," ensuring the model's memory stays pure during fine-tuning on isolated datasets.
* **📝 Automatic Instruction Tuning:** Built-in formatting that turns raw multi-column datasets into structured assistant prompts (Question/Answer/Reasoning).

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

### 1. Train the Brain 🏋️

Start training a model from scratch. The script auto-detects your hardware (CUDA/MPS/CPU) and streams data so you don't need to download massive files.
```bash
python train_hope.py
```
*   **Default Dataset:** `obekt/obekt-question-answer-reasoning-micro-v0.1` (~77K Q&A pairs with reasoning).
*   **Output:** Saves checkpoints to `hope_qa_micro.pth`.
*   **Dashboard:** Shows real-time Loss, Validation Loss, Speed (tok/s), and a Live Data Preview.
*   **Resume:** Training automatically resumes from the latest checkpoint if interrupted.

### 2. Quick Test 💬

Evaluate your model on sample questions:
```bash
python test_model.py
```

### 3. Chat in the Console 💬

Test your model immediately with a lightweight interactive chat optimized for speed.
```bash
python chat.py
```
*   Uses **Fast State-Passing** for $O(N)$ inference.
*   Shows real-time memory usage and parameter count.
*   Commands: `/temp 0.7`, `/tokens 250`, `/reasoning off`
*   Type `quit` to exit.

### 4. Run the Web Interface 🔌

Launch a beautiful Gradio-based web UI to chat with your model.
```bash
python app.py
```
*   **Inference Algorithm:** Optimized State-Passing ($O(N)$).
*   **Features:** Character-level streaming, temperature/top-p sliders, reasoning toggle.

### 5. Generate Non-Interactively 📝

```bash
python generate.py --prompt "What is AI?" --max-tokens 200 --temperature 0.7
```

---

## 🧪 Advanced Strategies

### 📈 Fine-Tuning
HOPE is natively designed for high-performance instruction tuning and domain adaptation.
1. **Prepare Data:** Use a Hugging Face dataset with columns like `question`, `answer`, and `reasoning`.
2. **Isolate Samples:** Set `"isolate_samples": True` in `CONFIG`. This ensures the model treats each row as a distinct fact, using **Loss Masking** to ignore padding.
3. **Auto-Formatting:** The trainer automatically detects multiple columns and formats them with headers (e.g., `Question: ... \nAnswer: ... \nReasoning: ...`), teaching the model assistant behaviors.
4. **Gentle Learning:** Use a lower `learning_rate` (e.g., `5e-5`) to refine the existing "Slow Weights" without losing the foundation knowledge.

### 🌡️ Inference Parameters
*   **Temperature:** Controls how "random" the model is.
    *   *Lower (0.1 - 0.5):* High confidence, strict logic.
    *   *Higher (0.8 - 1.2):* Creative, varied language.
*   **Top-p (Nucleus Sampling):** Only samples from the smallest set of tokens whose cumulative probability exceeds p. Reduces nonsense at high temperatures.
*   **Max Tokens:** Safety limit for generation. Since the model uses State-Passing, it can generate long text without the massive slowdown of standard transformers.

---

## 🧪 Configuration

The project is currently tuned for a **~35M Parameter model** optimized for the Q&A micro dataset. You can tweak the model size in `train_hope.py` by modifying the `CONFIG` dictionary:

```python
CONFIG = {
    "d_model": 512,           # Width (Reasoning capability)
    "n_layers": 16,           # Depth
    "seq_len": 512,           # Training window
    "vocab_size": 256,        # Byte-Level
    "max_steps": 3000,        # Training steps
    "learning_rate": 1e-3,    # Learning rate
    "isolate_samples": True,  # True for Q&A datasets, False for Wikipedia
}
```

### 🧠 Performance & RAM Specs
One of the key strengths of this architecture is its efficiency during use:

*   **Training (`train_hope.py`):** Uses ~8GB - 16GB RAM depending on config.
*   **Inference (`chat.py` / `app.py`):** Uses < 1GB RAM. Because of our **Fast State-Passing** optimization, the model only needs to remember its current state, making it incredibly lightweight for daily use.

### Preset Configurations:
- **Nano**: `d_model=256, n_layers=4` (Fastest, ~10M params)
- **Balanced**: `d_model=384, n_layers=12` (~50M params)
- **Deep**: `d_model=384, n_layers=32` (~100M params)
- **Ultra**: `d_model=768, n_layers=32` (~154M params, high-end Mac/PC)
- **QA Micro (Default)**: `d_model=512, n_layers=16` (~35M params, good for Q&A)

---

## 🧪 Training Laboratory & Experiments

The project was developed through a structured multi-phase experimental roadmap:

### Phase 1: General Foundation (Grammar & Facts)
*   **Dataset:** `wikimedia/wikipedia` (English)
*   **Method:** **Packed Training** (Stitching articles together to maximize density).
*   **Goal:** Building a deep semantic understanding of English and general knowledge.

### Phase 2: Structural Optimization (Performance)
*   **Inference:** Switched from $O(N^2)$ to **$O(N)$ State-Passing**. This enabled instant responses by carrying the memory matrix forward rather than re-calculating the entire sequence.
*   **Dataset Loader:** Upgraded to a **Token Buffer** system, ensuring 100% data utilization by eliminating stub-article discarding.

### Phase 3: Instruction Fine-Tuning (Critical Discovery)
*   **Dataset:** `obekt/obekt-question-answer-reasoning-micro-v0.1`
*   **Method:** **Sample-Isolated Mode** with **Padding Masking**.
*   **The "Padding Leakage" Discovery:** We found that without masking, the model's self-modifying memory would update during padding zeros, causing factual blending.
*   **The Fix:** Implemented a binary mask in the architecture. The memory now stays perfectly locked during padding, enabling pure, focused learning.
*   **Optimal Training Window:** For small datasets, **5-10 epochs** is the sweet spot. Over-training leads to "thematic blending" and hallucinations.

### 💡 High-Quality Best Practices
1.  **Always Mask:** Ensure `isolate_samples` is True when using Q&A data to trigger the binary padding mask.
2.  **Watch the Epochs:** Do not let the model see the same small dataset more than 15 times unless knowledge acquisition is still actively improving.
3.  **Instruction Template:** Always use the `Question: / Answer:` template in inference to match the model's fine-tuned state.

---

## 📜 Credits & Citation

This code is an unofficial implementation and experimental exploration of the concepts introduced in:

> **Nested Learning: The Illusion of Deep Learning**  
> Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research)  
> [Paper Link](https://abehrouz.github.io/files/NL.pdf)
