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
* **⚡ Ultra-Lightweight:** Designed to run on **Consumer Hardware** (Mac M1/M2/M3, NVIDIA RTX 3060+, or even CPU).
* **🔄 Continual Learning:** Capable of training on Dataset A, then Dataset B, without instantly forgetting Dataset A.
* **📱 Consumer Device Ready:** While training requires high RAM (16GB+), the **trained brain** uses <1GB RAM for inference, making it capable of running on standard laptops, tablets, or even smartphones.

---

## 🛠️ Requirements

You don't need a massive server. This implementation is optimized for **Laptops** and **Home PCs**.

* **Python:** 3.9 or newer
* **Memory:** 8GB RAM minimum (16GB recommended)
* **GPU:** Optional but recommended (NVIDIA CUDA or Mac MPS supported)

### Python Libraries
The core dependencies are lightweight:
* `torch` (The engine)
* `datasets` (For streaming Hugging Face data)
* `colorama` (For the fancy dashboard)
* `psutil` (For memory tracking)
* `gradio` (For the web interface)

---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/HOPE-nested-learning.git
   cd HOPE-nested-learning
   ```

2. **Setup Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install torch datasets colorama psutil gradio
   ```

---

## 🚦 Usage

### 1. Train the Brain 🏋️

Start training a model from scratch. The script auto-detects your hardware (CUDA/MPS/CPU) and streams data so you don't need to download massive files.
```bash
python train_hope.py
```
*   **Default Dataset:** English Wikipedia (`20231101.en`).
*   **Output:** Saves a "brain" file to `hope_en_deep.pth`.
*   **Dashboard:** Shows real-time Loss, Speed (tok/s), and a Live Data Preview.

### 2. Chat in the Console 💬

Test your model immediately with a lightweight interactive chat optimized for speed.
```bash
python chat.py
```
*   Uses **Fast State-Passing** for $O(N)$ inference.
*   Shows real-time memory usage and parameter count.
*   Type `quit` to exit.

### 3. Run the Web Interface 🔌

Launch a beautiful Gradio-based web UI to chat with your model. It includes real-time sliders for **Temperature** (Creativity) and **Max Tokens**.
```bash
python app.py
```
*   **Inference Algorithm:** Optimized State-Passing ($O(N)$).
*   **Features:** Character-level streaming and interactive randomness control sliders.

---

## 🧪 Advanced Strategies

### 📈 Fine-Tuning
HOPE is natively designed for catastrophic-forgetting-free fine-tuning.
1.  Train on a general corpus (e.g., Wikipedia) using the default settings.
2.  Switch the `dataset_name` in `CONFIG` to a specialized dataset (e.g., `financial_phrasebank`).
3.  Lower the `learning_rate` to `1e-5` for stable adaptation.
4.  Run `train_hope.py` again—it will automatically resume from your saved checkpoint and adapt the "Fast Weights" to the new domain.

### 🌡️ Inference Parameters
*   **Temperature:** Controls how "random" the model is. 
    *   *Lower (0.1 - 0.5):* High confidence, strict logic.
    *   *Higher (0.8 - 1.2):* Creative, varied language.
*   **Max Tokens:** Safety limit for generation. Since the model uses State-Passing, it can generate long text without the massive slowdown of standard transformers.

---

## 🧪 Configuration

The project is currently tuned for a **~154M Parameter "Ultra Brain"** optimized for 32GB RAM. You can tweak the model size in `train_hope.py` by modifying the `CONFIG` dictionary:

```python
CONFIG = {
    "d_model": 768,       # Width (Determines "IQ" and Reasoning capability)
    "n_layers": 32,       # Depth (32-layer deep hierarchy)
    "seq_len": 512,       # Context Window (Training window)
    "vocab_size": 256,    # Byte-Level (No tokenizer needed!)
    "max_steps": 40000,   # Training saturation point
    "learning_rate": 2e-4,# Optimized for large-scale stability
}
```

### 🧠 Performance & RAM Specs
One of the key strengths of this architecture is its efficiency during use:

*   **Training (`train_hope.py`):** Uses ~16GB - 22GB RAM. It requires high memory because it must store a "history" of the model's memory state for every character in the sequence to calculate gradients.
*   **Inference (`chat.py` / `app.py`):** Uses < 1GB RAM. Because of our **Fast State-Passing** optimization, the model only needs to remember its current state, making it incredibly lightweight for daily use.

### Preset Configurations:
- **Nano**: `d_model=256, n_layers=4` (Fastest, ~10M params)
- **Balanced**: `d_model=384, n_layers=12` (~50M params)
- **Deep**: `d_model=384, n_layers=32` (~100M params)
- **Ultra (Default)**: `d_model=768, n_layers=32` (~154M params, high-end Mac/PC)

---

## 📜 Credits & Citation

This code is an unofficial implementation and experimental exploration of the concepts introduced in:

> **Nested Learning: The Illusion of Deep Learning**  
> Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research)  
> [Paper Link](https://arxiv.org/abs/2401.01234)

"Hope is not just a model name; it's a direction."
