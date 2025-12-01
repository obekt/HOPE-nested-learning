import torch
import gradio as gr
from train_hope import HOPE, CONFIG, DEVICE  # Import your model definition

# 1. Load the Model
print("Loading HOPE Model...")
model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers'])
model.load_state_dict(torch.load("hope_financial_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Simple decoder (Byte-level)
def decode(tokens):
    return bytes(tokens).decode('utf-8', errors='ignore')

def predict(message, history):
    # 2. Prepare Input
    # Convert text to bytes (0-255)
    input_ids = list(message.encode('utf-8'))
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    # 3. Generate (Simple loop)
    generated = []
    
    # Reset Fast Memory for the new prompt (Crucial for HOPE)
    # Note: In a full implementation, you'd integrate the reset logic here
    # model.fast_memory.reset_state() 
    
    for _ in range(100): # Generate 100 tokens max
        with torch.no_grad():
            logits = model(x)
            
        # Get last token prediction
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to input for next step (Auto-regressive)
        x = torch.cat((x, next_token), dim=1)
        
        token_int = next_token.item()
        generated.append(token_int)
        
        # Stop if we hit a specific stop token (optional)
        if token_int == 0: break 
        
        # Stream the output to the UI
        yield decode(generated)

# 4. Launch the "Mini Studio"
demo = gr.ChatInterface(
    predict,
    title="HOPE Financial Model (Experimental)",
    description="A 50M parameter Nested Learning model trained on financial news.",
    examples=["Market outlook for", "The stock price of"],
    type="messages" # Use standard chat format
)

if __name__ == "__main__":
    demo.launch(share=True) # share=True creates a public link!