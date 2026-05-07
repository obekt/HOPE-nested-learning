import torch
import gradio as gr
import os
import sys

# Import your model definition from the training script
from train_hope import HOPE, CONFIG, DEVICE

# --- CONFIGURATION ---
MODEL_FILENAME = CONFIG['save_path']
BEST_MODEL_FILENAME = MODEL_FILENAME.replace('.pth', '_best.pth')

print(f"Loading HOPE Model from {MODEL_FILENAME}...")

# 1. Initialize the empty model architecture
try:
    model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers'])
except NameError:
    print("Error: Could not find HOPE class. Make sure train_hope.py is in the same folder.")
    sys.exit(1)

# 2. Determine which checkpoint to load
load_path = MODEL_FILENAME
if os.path.exists(BEST_MODEL_FILENAME):
    if not os.path.exists(MODEL_FILENAME):
        load_path = BEST_MODEL_FILENAME
        print(f"Best checkpoint found: {BEST_MODEL_FILENAME}")
    else:
        # Load whichever is newer
        if os.path.getmtime(BEST_MODEL_FILENAME) > os.path.getmtime(MODEL_FILENAME):
            load_path = BEST_MODEL_FILENAME
            print(f"Loading newer best checkpoint: {BEST_MODEL_FILENAME}")

if not os.path.exists(load_path):
    print(f"Error: No model checkpoint found at {load_path}!")
    print("Please train a model first with: python train_hope.py")
    sys.exit(1)

checkpoint = torch.load(load_path, map_location=DEVICE)

# 3. UNPACK SMART CHECKPOINT
if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    print("Detected Smart Checkpoint. Unpacking weights...")
    state_dict = checkpoint['model_state']
else:
    print("Detected Legacy Checkpoint.")
    state_dict = checkpoint

# 4. Load weights into model
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")


# --- CHAT LOGIC ---

def decode_tokens(tokens):
    """Helper to convert list of numbers to string"""
    return bytes(tokens).decode('utf-8', errors='ignore')


def predict(message, history, temperature, max_tokens, top_p, show_reasoning):
    if not message.strip():
        return

    # Format prompt to match training template
    full_prompt = f"Question: {message.strip()}\nAnswer: "

    # 1. Encode Input
    input_ids = list(full_prompt.encode('utf-8'))
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    generated_ids = []

    # First Pass: Process the whole prompt to build initial state
    with torch.no_grad():
        logits, state = model(x)

    last_token_logits = logits[:, -1, :] / max(0.01, temperature)

    # Top-p sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[0][sorted_indices_to_remove[0]]
        last_token_logits[0, indices_to_remove] = -float('Inf')

    probs = torch.softmax(last_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    reasoning_seen = False
    output_buffer = ""

    # 2. Optimized Generation Loop
    for _ in range(max_tokens):
        token_int = next_token.item()

        if token_int == 0:
            break

        generated_ids.append(token_int)

        text = decode_tokens(generated_ids)

        # Optionally hide reasoning section
        if not show_reasoning and "Reasoning:" in text:
            parts = text.split("Reasoning:")
            if len(parts) > 1:
                text = parts[0].strip()
                break

        yield text

        with torch.no_grad():
            x = next_token
            logits, state = model(x, state=state)

        last_token_logits = logits[:, -1, :] / max(0.01, temperature)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[0][sorted_indices_to_remove[0]]
            last_token_logits[0, indices_to_remove] = -float('Inf')

        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)


# --- LAUNCH UI ---
demo = gr.ChatInterface(
    predict,
    additional_inputs=[
        gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature (Creativity)"),
        gr.Slider(minimum=50, maximum=1000, value=300, step=50, label="Max Tokens (Length)"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p (Nucleus Sampling)"),
        gr.Checkbox(value=True, label="Show Reasoning")
    ],
    title="HOPE: Nested Learning Chat (Q&A Micro)",
    description=f"Running {os.path.basename(load_path)} on {DEVICE}",
    examples=[
        ["What is artificial intelligence?", 0.7, 300, 0.9, True],
        ["Tell me a story about a robot.", 0.8, 300, 0.9, True],
        ["Explain quantum computing in simple terms.", 0.7, 300, 0.9, True],
        ["Why is the sky blue?", 0.6, 200, 0.9, True],
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
