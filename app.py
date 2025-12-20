import torch
import gradio as gr
import os
import sys

# Import your model definition from the training script
from train_hope import HOPE, CONFIG, DEVICE

# --- CONFIGURATION ---
# --- CONFIGURATION ---
MODEL_FILENAME = "hope_granite_hybrid.pth"  # Updated for Granite hybrid model

print(f"Loading HOPE Model from {MODEL_FILENAME}...")

# 1. Initialize the empty model architecture
try:
    model = HOPE(CONFIG['granite_model'], CONFIG['d_model'], CONFIG['n_layers'])
except NameError:
    print("Error: Could not find HOPE class. Make sure train_hope.py is in the same folder.")
    sys.exit(1)

# 2. Load the file (if exists, else it will just be the backbone)
if os.path.exists(MODEL_FILENAME):
    checkpoint = torch.load(MODEL_FILENAME, map_location=DEVICE)

    # 3. UNPACK SMART CHECKPOINT
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        print("Detected Smart Checkpoint. Unpacking weights...")
        state_dict = checkpoint['model_state']
    else:
        print("Detected Legacy Checkpoint.")
        state_dict = checkpoint

    # 4. Load weights into model
    model.load_state_dict(state_dict, strict=False)
else:
    print(f"No checkpoint found at {MODEL_FILENAME}. Using raw Granite backbone + uninitialized HOPE layers.")

# --- OPTIMIZATION: Use Half Precision (FP16) for Mac/MPS and CUDA ---
if DEVICE in ["mps", "cuda"]:
    print(f"Optimizing for {DEVICE} (Half Precision)...")
    model = model.half()

model.to(DEVICE)
model.eval()
print("Model ready!")

import time

# --- CHAT LOGIC ---

def decode_tokens(tokens, tokenizer):
    """Helper to convert list of numbers to string"""
    return tokenizer.decode(tokens)

def predict(message, history):
    # 1. Encode Input using Granite tokenizer
    tokenizer = model.granite.tokenizer
    input_ids = tokenizer.encode(message, add_special_tokens=False)
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    generated_ids = []
    start_time = time.time()
    
    # 2. Generation Loop
    for _ in range(256): # Max 256 tokens
        # Prepare tokens_dict for Granite-based forward pass
        tokens_dict = {
            'input_ids': x,
            'attention_mask': torch.ones_like(x).to(DEVICE)
        }
        
        with torch.inference_mode():
            logits = model(tokens_dict)
            
        # Sampling
        last_token_logits = logits[:, -1, :]
        # Temperature control (hardcoded to 0.7 for consistency)
        probs = torch.softmax(last_token_logits / 0.7, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append
        x = torch.cat((x, next_token), dim=1)
        token_int = next_token.item()
        
        if token_int == tokenizer.eos_token_id: break # Stop token
        
        generated_ids.append(token_int)
        
        # Calculate stats
        elapsed = time.time() - start_time
        tps = len(generated_ids) / elapsed if elapsed > 0 else 0
        
        # Stream output to UI with statistics appended in a small footer
        output_text = decode_tokens(generated_ids, tokenizer)
        stats_footer = f"\n\n---\n*⚡ {len(generated_ids)} tokens | {tps:.1f} tok/s*"
        yield output_text + stats_footer

with gr.Blocks(title="HOPE Hybrid Chat") as demo:
    gr.Markdown(f"""
    # 🧠 HOPE: Hybrid Nested Learning
    **Architecture:** {CONFIG['n_layers']} Layers | **Backbone:** IBM Granite English
    **Device:** {DEVICE} | **Model:** `{MODEL_FILENAME}`
    """)
    
    chat = gr.ChatInterface(
        predict,
        examples=[
            "Explain the concept of Nested Learning.",
            "Tell me about the importance of IBM Granite embeddings.",
            "Write a short poem about artificial intelligence.",
            "How does the Fast Memory layer work in this model?"
        ]
    )
    
    gr.Markdown("""
    ---
    *Note: This is an experimental hybrid model combining static semantic foundations with real-time adaptive weights.*
    """)

if __name__ == "__main__":
    demo.launch(share=True)
