#!/usr/bin/env python3
"""
Non-interactive text generation with HOPE.
Usage:
    python generate.py --prompt "What is AI?" --max-tokens 200 --temperature 0.7
"""
import torch
import torch.nn.functional as F
import argparse
import sys
import os
from train_hope import HOPE, CONFIG, DEVICE


def load_model(path):
    model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers'])
    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def generate(model, prompt, max_new_tokens=200, temperature=0.7):
    full_prompt = f"Question: {prompt.strip()}\nAnswer: "
    input_ids = list(full_prompt.encode('utf-8'))
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    generated = []

    with torch.no_grad():
        logits, state = model(x)

    last_token_logits = logits[:, -1, :] / max(0.01, temperature)
    probs = F.softmax(last_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    for _ in range(max_new_tokens):
        token_int = next_token.item()
        if token_int == 0:
            break
        generated.append(token_int)

        with torch.no_grad():
            x = next_token
            logits, state = model(x, state=state)

        last_token_logits = logits[:, -1, :] / max(0.01, temperature)
        probs = F.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    return bytes(generated).decode('utf-8', errors='ignore')


def main():
    parser = argparse.ArgumentParser(description="Generate text with HOPE")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt/question")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    model_path = args.model or CONFIG['save_path']
    best_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_path) and not os.path.exists(model_path):
        model_path = best_path

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train first with: python train_hope.py")
        sys.exit(1)

    model = load_model(model_path)
    print(f"Generating with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {DEVICE}\n")
    print(f"Prompt: {args.prompt}\n")
    print("-" * 40)

    result = generate(model, args.prompt, args.max_tokens, args.temperature)
    print(result)


if __name__ == "__main__":
    main()
