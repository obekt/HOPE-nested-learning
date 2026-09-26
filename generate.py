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
from train_hope import (
    HOPE, CONFIG, DEVICE, TOKENIZER, EOS_TOKEN_ID,
    load_model_for_inference, sample_next_token, maybe_compile_for_steps,
)


def load_model(path, dtype=torch.bfloat16):
    """Shared strict loader (strict=True inside load_model_for_inference)."""
    model, _ = load_model_for_inference(path, device=DEVICE, dtype=dtype)
    return model


def generate(model, prompt, max_new_tokens=200, temperature=0.7,
             repetition_penalty=1.15, step_model=None):
    full_prompt = f"Question: {prompt.strip()}\nAnswer: "
    input_ids = TOKENIZER.encode(full_prompt, return_tensors="pt").to(DEVICE)
    step_model = step_model or model

    generated = []
    prev_tokens = input_ids[0].tolist()

    with torch.no_grad():
        # last_only: CMS stack + head on the final position only (2.3x prefill)
        logits, state = model(input_ids, last_only=True)

    next_token = sample_next_token(logits, temperature=temperature,
                                   repetition_penalty=repetition_penalty,
                                   prev_tokens=prev_tokens)

    for _ in range(max_new_tokens):
        token_int = next_token.item()
        if token_int == EOS_TOKEN_ID:
            break
        generated.append(token_int)
        prev_tokens.append(token_int)

        with torch.no_grad():
            logits, state = step_model(next_token, state=state, last_only=True)

        next_token = sample_next_token(logits, temperature=temperature,
                                       repetition_penalty=repetition_penalty,
                                       prev_tokens=prev_tokens)

    return TOKENIZER.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text with HOPE")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt/question")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                        help="bf16 (default): faster, validated quality-identical; fp32: legacy")
    parser.add_argument("--repetition-penalty", type=float, default=1.15,
                        help="1.0 disables; 1.1-1.3 tames repetition loops")
    parser.add_argument("--no-compile", action="store_true",
                        help="disable torch.compile for generation steps")
    args = parser.parse_args()

    model_path = args.model or CONFIG['save_path']
    best_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_path) and not os.path.exists(model_path):
        model_path = best_path

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train first with: python train_hope.py")
        sys.exit(1)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = load_model(model_path, dtype=dtype)
    step_model = model if args.no_compile else maybe_compile_for_steps(model)
    print(f"Generating with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {DEVICE} ({args.dtype})\n")
    print(f"Prompt: {args.prompt}\n")
    print("-" * 40)

    result = generate(model, args.prompt, args.max_tokens, args.temperature,
                      repetition_penalty=args.repetition_penalty, step_model=step_model)
    print(result)


if __name__ == "__main__":
    main()
