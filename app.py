import torch
import gradio as gr
import os
import sys

from train_hope import (
    HOPE, CONFIG, DEVICE, TOKENIZER, EOS_TOKEN_ID,
    load_model_for_inference, sample_next_token, maybe_compile_for_steps,
)

MODEL_FILENAME = CONFIG['save_path']
BEST_MODEL_FILENAME = MODEL_FILENAME.replace('.pth', '_best.pth')

print(f"Loading HOPE Model from {MODEL_FILENAME}...")

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

try:
    # bf16 weights (validated quality-identical); shared loader is strict=True
    model, checkpoint = load_model_for_inference(load_path, device=DEVICE, dtype=torch.bfloat16)
except RuntimeError as e:
    print(f"Architecture Mismatch! Your saved model differs from CONFIG. Details: {e}")
    sys.exit(1)

step_model = maybe_compile_for_steps(model)
print(f"Model loaded successfully! ({load_path})")


def predict(message, history, temperature, max_tokens, show_reasoning):
    if not message.strip():
        return

    full_prompt = f"Question: {message.strip()}\nAnswer: "
    input_ids = TOKENIZER.encode(full_prompt, return_tensors="pt").to(DEVICE)
    generated_ids = []
    prev_tokens = input_ids[0].tolist()

    with torch.no_grad():
        logits, state = model(input_ids, last_only=True)

    next_token = sample_next_token(logits, temperature=temperature, prev_tokens=prev_tokens)

    for _ in range(max_tokens):
        token_int = next_token.item()
        if token_int == EOS_TOKEN_ID:
            break
        generated_ids.append(token_int)
        prev_tokens.append(token_int)

        text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)
        if not show_reasoning and "Reasoning:" in text:
            parts = text.split("Reasoning:")
            if len(parts) > 1:
                text = parts[0].strip()
                yield text
                break
        yield text

        with torch.no_grad():
            logits, state = step_model(next_token, state=state, last_only=True)

        next_token = sample_next_token(logits, temperature=temperature, prev_tokens=prev_tokens)


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
