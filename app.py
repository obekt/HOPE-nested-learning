import torch
import gradio as gr
import os
import sys

from train_hope import HOPE, CONFIG, DEVICE, TOKENIZER, EOS_TOKEN_ID

MODEL_FILENAME = CONFIG['save_path']
BEST_MODEL_FILENAME = MODEL_FILENAME.replace('.pth', '_best.pth')

print(f"Loading HOPE Model from {MODEL_FILENAME}...")

try:
    model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers'])
except NameError:
    print("Error: Could not find HOPE class. Make sure train_hope.py is in the same folder.")
    sys.exit(1)

load_path = MODEL_FILENAME
if os.path.exists(BEST_MODEL_FILENAME):
    if not os.path.exists(MODEL_FILENAME):
        load_path = BEST_MODEL_FILENAME
    else:
        if os.path.getmtime(BEST_MODEL_FILENAME) > os.path.getmtime(MODEL_FILENAME):
            load_path = BEST_MODEL_FILENAME

if not os.path.exists(load_path):
    print(f"Error: No model checkpoint found at {load_path}!")
    print("Please train a model first with: python train_hope.py")
    sys.exit(1)

checkpoint = torch.load(load_path, map_location=DEVICE)

if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    print("Detected Smart Checkpoint. Unpacking weights...")
    state_dict = checkpoint['model_state']
else:
    print("Detected Legacy Checkpoint.")
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")


def predict(message, history, temperature, max_tokens, show_reasoning):
    if not message.strip():
        return

    full_prompt = f"Question: {message.strip()}\nAnswer: "
    input_ids = TOKENIZER.encode(full_prompt, return_tensors="pt").to(DEVICE)
    generated_ids = []

    with torch.no_grad():
        logits, state = model(input_ids)

    last_token_logits = logits[:, -1, :] / max(0.01, temperature)
    probs = torch.softmax(last_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    for _ in range(max_tokens):
        token_int = next_token.item()
        if token_int == EOS_TOKEN_ID:
            break
        generated_ids.append(token_int)

        text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)
        if not show_reasoning and "Reasoning:" in text:
            parts = text.split("Reasoning:")
            if len(parts) > 1:
                text = parts[0].strip()
                yield text
                break
        yield text

        with torch.no_grad():
            x = next_token
            logits, state = model(x, state=state)

        last_token_logits = logits[:, -1, :] / max(0.01, temperature)
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)


demo = gr.ChatInterface(
    predict,
    additional_inputs=[
        gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature (Creativity)"),
        gr.Slider(minimum=50, maximum=1000, value=300, step=50, label="Max Tokens (Length)"),
        gr.Checkbox(value=True, label="Show Reasoning")
    ],
    title="HOPE: Nested Learning Chat (GPT-2 Tokenizer)",
    description=f"Running {os.path.basename(load_path)} on {DEVICE}",
    examples=[
        ["What is artificial intelligence?", 0.7, 300, True],
        ["Tell me a story about a robot.", 0.8, 300, True],
        ["Explain quantum computing in simple terms.", 0.7, 300, True],
        ["Why is the sky blue?", 0.6, 200, True],
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
