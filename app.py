import torch
import gradio as gr
import os
import sys

# Import your model definition from the training script
from train_hope import HOPE, CONFIG, DEVICE

# --- CONFIGURATION ---
MODEL_FILENAME = "hope_en_deep.pth"  # Updated for English model

print(f"Loading HOPE Model from {MODEL_FILENAME}...")

# 1. Initialize the empty model architecture
try:
    model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers'])
except NameError:
    print("Error: Could not find HOPE class. Make sure train_hope.py is in the same folder.")
    sys.exit(1)

# 2. Load the file
if not os.path.exists(MODEL_FILENAME):
    print(f"Error: {MODEL_FILENAME} not found!")
    sys.exit(1)

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
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# --- CHAT LOGIC ---

def decode_tokens(tokens):
    """Helper to convert list of numbers to string"""
    return bytes(tokens).decode('utf-8', errors='ignore')

def predict(message, history):
    # Note: 'history' is passed by Gradio but we aren't using it in this simple version
    # If you want memory, you would concatenate history string here.
    
    # 1. Encode Input
    input_ids = list(message.encode('utf-8'))
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    generated_ids = []
    
    # 2. Generation Loop
    for _ in range(200): # Max 200 tokens
        with torch.no_grad():
            logits = model(x)
            
        # Sampling
        last_token_logits = logits[:, -1, :]
        # Temperature control (hardcoded to 0.6 for stability)
        probs = torch.softmax(last_token_logits / 0.6, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append
        x = torch.cat((x, next_token), dim=1)
        token_int = next_token.item()
        
        if token_int == 0: break # Stop token
        
        generated_ids.append(token_int)
        
        # Stream output to UI
        yield decode_tokens(generated_ids)

# --- LAUNCH UI ---
demo = gr.ChatInterface(
    predict,
    title="HOPE: Nested Learning Chat (English)",
    description=f"Running {MODEL_FILENAME} on {DEVICE}",
    examples=[
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Tell me a story about a robot.",
        "Explain quantum computing in simple terms."
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
