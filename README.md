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
* **🕰️ Continuum Memory System (CMS):** A hierarchy of layers that update at different frequencies (Fast, Medium, Slow), mimicking the human brain's memory consolidation.
* **⚡ Ultra-Lightweight:** Designed to run on **Consumer Hardware** (Mac M1/M2/M3, NVIDIA RTX 3060+, or even CPU).
* **🔄 Continual Learning:** Capable of training on Dataset A, then Dataset B, without instantly forgetting Dataset A.

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
* `fastapi` & `uvicorn` (Only if you want to run the local API server)

---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/HOPE-nested-learning.git](https://github.com/YOUR_USERNAME/HOPE-nested-learning.git)
   cd HOPE-nested-learning

    Install Dependencies
    Bash

    pip install torch datasets colorama psutil

    # Optional: For the API Server
    pip install fastapi uvicorn

🚦 Usage

1. Train the Brain 🏋️

Start training a model from scratch. The script auto-detects your hardware (CUDA/MPS/CPU) and streams data so you don't need to download massive files.
Bash

    python train_hope.py

Default Dataset: English Wikipedia (20231101.en).

Output: Saves a "brain" file to hope_en_deep.pth.

Dashboard: Shows real-time Loss, Speed (tok/s), and a Live Data Preview.

2. Chat in the Console 💬

Test your model immediately with a lightweight interactive chat.
Bash

    python chat.py

Shows real-time memory usage and parameter count.

Type quit to exit.

3. Connect to LM Studio / Web UI 🔌

Want to use a nice UI? Run the API server, which mimics OpenAI's API.
Bash

    python app.py

Endpoint: http://localhost:8000

Compatible with: LM Studio, Chatbox AI, SillyTavern, etc.

🧪 Configuration

You can tweak the model size in train_hope.py by modifying the CONFIG dictionary:
Python

CONFIG = {
    "d_model": 384,       # Width (Increased for better understanding)
    "n_layers": 32,       # Depth (Deep architecture for complex patterns)
    "seq_len": 768,       # Context Window (Longer for better comprehension)
    "vocab_size": 256,    # Byte-Level (No tokenizer needed!)
    "max_steps": 15000,   # Training steps
    "learning_rate": 3e-4,# Optimized for deep networks
    "warmup_steps": 500,  # Learning rate warmup
}

**Preset Configurations:**

- **Nano Mode**: d_model=256, n_layers=4, seq_len=512 (Fastest, good for testing)
- **Balanced**: d_model=256, n_layers=12, seq_len=512 (Good for laptops)
- **Deep (Default)**: d_model=384, n_layers=32, seq_len=768 (Best quality, requires 16GB RAM)

**Enhanced Features:**
- ✅ Learning rate warmup for stable training
- ✅ Gradient accumulation for effective larger batch sizes
- ✅ Automatic checkpoint saving with optimizer state
- ✅ Real-time training dashboard with live data preview

📜 Credits & Citation

This code is an unofficial implementation and experimental exploration of the concepts introduced in:

Nested Learning: The Illusion of Deep Learning > Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research) > Paper Link

"Hope is not just a model name; it's a direction."
