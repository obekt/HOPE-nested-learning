import torch
import torch.nn.functional as F
import sys
import os
import psutil
from colorama import Fore, Style, init

from train_hope import (
    HOPE, CONFIG, DEVICE, TOKENIZER, PAD_TOKEN_ID, EOS_TOKEN_ID,
    load_model_for_inference, sample_next_token, maybe_compile_for_steps,
    uses_qa_format, wrap_prompt,
)

init(autoreset=True)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mb = process.memory_info().rss / 1024 / 1024
    return mb


def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    current_ram = get_memory_usage()

    print(f"\n{Fore.GREEN}=== MODEL STATISTICS ==={Style.RESET_ALL}")
    print(f"Architecture:   HOPE (Nested Learning)")
    print(f"Parameters:     {Fore.YELLOW}{total_params / 1_000_000:.2f} Million{Style.RESET_ALL}")
    size_on_disk = (total_params * 4) / 1024 / 1024
    print(f"Est. File Size: {size_on_disk:.1f} MB")
    print(f"Active Memory:  {Fore.CYAN}{current_ram:.1f} MB{Style.RESET_ALL}")
    if DEVICE == "cuda":
        vram = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU VRAM:       {vram:.1f} MB")
    print("=" * 30 + "\n")


def load_model(path):
    """Loads via the shared strict loader (strict=True inside
    load_model_for_inference). bf16 by default — pass --fp32 for legacy."""
    print(f"{Fore.YELLOW}Loading model from {path}...{Style.RESET_ALL}")
    dtype = torch.float32 if "--fp32" in sys.argv else torch.bfloat16
    try:
        model, checkpoint = load_model_for_inference(path, device=DEVICE, dtype=dtype)
        if checkpoint is not None:
            print(f"{Fore.CYAN}Detected Smart Checkpoint (step {checkpoint.get('step', 'unknown')}). Unpacking...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Precision: {dtype} | scan/memory state: fp32{Style.RESET_ALL}")
        return model, checkpoint

    except FileNotFoundError:
        print(f"{Fore.RED}Error: Model file '{path}' not found.{Style.RESET_ALL}")
        print(f"Current Config points to: {CONFIG['save_path']}")
        print(f"\n{Fore.YELLOW}Hint: Run training first with 'python train_hope.py'{Style.RESET_ALL}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"{Fore.RED}Architecture Mismatch!{Style.RESET_ALL}")
        print("Your saved model has different dimensions than your current CONFIG.")
        print(f"Error details: {e}")
        sys.exit(1)


# Wikipedia-style back-matter: everything after these markers is References /
# category-stub filler, so raw-mode (foundation) generation stops there.
END_OF_ARTICLE_MARKERS = (
    "\n\nReferences", "\n\nExternal links", "\n\nSee also",
    "\n\nFurther reading", "\n\nBibliography", "\n\nSources",
    "\n\nNotes",
)
_STREAM_HOLDBACK = 24  # chars withheld so a half-typed marker is never printed


def generate_response(model, prompt, max_new_tokens=250, temperature=0.7, show_reasoning=True, step_model=None, qa_format=True):
    full_prompt = wrap_prompt(prompt, qa_format)
    input_ids = TOKENIZER.encode(full_prompt, return_tensors="pt").to(DEVICE)
    step_model = step_model or model

    print(f"\n{Fore.CYAN}HOPE: {Style.RESET_ALL}", end="", flush=True)

    generated_ids = []
    prev_tokens = input_ids[0].tolist()  # repetition-penalty context
    shown_upto = 0   # incremental streaming: chars already on screen
    final_text = ""

    with torch.no_grad():
        # last_only: CMS stack + head run on the final position only (2.3x prefill)
        logits, state = model(input_ids, last_only=True)

    next_token = sample_next_token(logits, temperature=temperature, prev_tokens=prev_tokens)

    stop = False
    for _ in range(max_new_tokens):
        token_int = next_token.item()
        if token_int == EOS_TOKEN_ID:
            break
        generated_ids.append(token_int)
        prev_tokens.append(token_int)

        text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)

        if not show_reasoning and "Reasoning:" in text:
            text = text.split("Reasoning:")[0].strip()
            stop = True
        elif not qa_format:
            # foundation model: cut before the end-of-article back-matter
            cut = -1
            for marker in END_OF_ARTICLE_MARKERS:
                i = text.find(marker)
                if i != -1 and (cut == -1 or i < cut):
                    cut = i
            if cut != -1:
                text = text[:cut].rstrip()
                stop = True

        final_text = text
        # Print only what's NEW since the last token. The old approach
        # re-printed the full text with \r every token — once a line wrapped,
        # \r couldn't overwrite it and the terminal flooded with copies.
        safe = len(text) if stop else max(0, len(text) - _STREAM_HOLDBACK)
        if safe > shown_upto:
            print(text[shown_upto:safe], end="", flush=True)
            shown_upto = safe

        if stop:
            break

        with torch.no_grad():
            logits, state = step_model(next_token, state=state, last_only=True)

        next_token = sample_next_token(logits, temperature=temperature, prev_tokens=prev_tokens)

    # flush the held-back tail
    if len(final_text) > shown_upto:
        print(final_text[shown_upto:], end="", flush=True)
    print()
    return final_text


def main():
    # --model <path> overrides CONFIG['save_path'] (argv-style like --fp32)
    model_path = CONFIG['save_path']
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
        else:
            print(f"{Fore.RED}--model requires a path{Style.RESET_ALL}")
            sys.exit(1)
    best_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_path) and not os.path.exists(model_path):
        model_path = best_path
        print(f"{Fore.CYAN}Using best checkpoint: {best_path}{Style.RESET_ALL}")

    model, checkpoint = load_model(model_path)
    print_model_stats(model)
    step_model = maybe_compile_for_steps(model)

    # Prompt format: auto-detect from the checkpoint's training-time flag
    # (or filename for older checkpoints). Foundation models degenerate to
    # whitespace under the Q&A wrapper; fine-tuned models need it.
    qa_format = uses_qa_format(model_path, checkpoint)
    print(f"{Fore.CYAN}Prompt format: {'Q&A wrapper' if qa_format else 'raw'} "
          f"(toggle with /format raw|qa){Style.RESET_ALL}")

    print("Interactive Console - Type 'quit' to exit")
    print("Commands: /temp <value>, /tokens <value>, /reasoning <on|off>, /format <raw|qa>")
    print("-" * 50)

    temperature = 0.7
    max_tokens = 250
    show_reasoning = True

    while True:
        try:
            user_input = input(f"\n{Fore.WHITE}You: {Style.RESET_ALL}")
            if not user_input.strip():
                continue

            if user_input.lower() in ["quit", "exit"]:
                break

            if user_input.startswith("/temp "):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except (ValueError, IndexError):
                    print("Usage: /temp 0.7")
                continue

            if user_input.startswith("/tokens "):
                try:
                    max_tokens = int(user_input.split()[1])
                    print(f"Max tokens set to {max_tokens}")
                except (ValueError, IndexError):
                    print("Usage: /tokens 250")
                continue

            if user_input.startswith("/reasoning "):
                val = user_input.split()[1].lower()
                show_reasoning = val in ("on", "true", "1", "yes")
                print(f"Reasoning display: {'on' if show_reasoning else 'off'}")
                continue

            if user_input.startswith("/format "):
                val = user_input.split()[1].lower()
                if val not in ("raw", "qa"):
                    print("Usage: /format raw  (verbatim prompt) or /format qa  (Question:/Answer: wrapper)")
                    continue
                qa_format = (val == "qa")
                print(f"Prompt format: {'Q&A wrapper' if qa_format else 'raw'}")
                continue

            try:
                generate_response(model, user_input, max_new_tokens=max_tokens, temperature=temperature, show_reasoning=show_reasoning, step_model=step_model, qa_format=qa_format)
            except KeyboardInterrupt:
                # Ctrl+C stops this response but keeps the chat alive
                print(f"\n{Fore.YELLOW}(generation stopped){Style.RESET_ALL}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
