#!/usr/bin/env python3
"""
Quick evaluation script for HOPE models.
Runs the model on a set of sample questions and prints outputs.
"""
import torch
import os
import sys
from train_hope import HOPE, CONFIG, DEVICE, TOKENIZER, EOS_TOKEN_ID


def load_model(path):
    model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers'])
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def generate(model, prompt, max_new_tokens=200, temperature=0.7):
    full_prompt = f"Question: {prompt.strip()}\nAnswer: "
    input_ids = TOKENIZER.encode(full_prompt, return_tensors="pt").to(DEVICE)

    generated = []

    with torch.no_grad():
        logits, state = model(input_ids)

    last_token_logits = logits[:, -1, :] / max(0.01, temperature)
    probs = torch.softmax(last_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    for _ in range(max_new_tokens):
        token_int = next_token.item()
        if token_int == EOS_TOKEN_ID:
            break
        generated.append(token_int)

        with torch.no_grad():
            x = next_token
            logits, state = model(x, state=state)

        last_token_logits = logits[:, -1, :] / max(0.01, temperature)
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    return TOKENIZER.decode(generated, skip_special_tokens=True)


TEST_QUESTIONS = [
    "Why is the sky blue?",
    "Can animals talk?",
    "What is artificial intelligence?",
    "Why do we sleep?",
    "What is the meaning of life?",
]


def main():
    model_path = CONFIG['save_path']
    best_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_path) and not os.path.exists(model_path):
        model_path = best_path

    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")
        print("Train first with: python train_hope.py")
        sys.exit(1)

    model = load_model(model_path)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {total_params/1e6:.1f}M parameters from {model_path}\n")
    print("=" * 60)

    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        print("-" * 40)
        answer = generate(model, question, max_new_tokens=200, temperature=0.7)
        print(f"A: {answer}")
        print("=" * 60)


if __name__ == "__main__":
    main()
