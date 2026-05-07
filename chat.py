import torch
import torch.nn.functional as F
import sys
import os
import psutil
from colorama import Fore, Style, init

# Import from your training script
from train_hope import HOPE, CONFIG, DEVICE

# Initialize colors
init(autoreset=True)


def get_memory_usage():
    """Returns the RAM usage of the current Python process in MB."""
    process = psutil.Process(os.getpid())
    mb = process.memory_info().rss / 1024 / 1024
    return mb


def print_model_stats(model):
    """Calculates and prints model size and memory footprint."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    current_ram = get_memory_usage()

    print(f"\n{Fore.GREEN}=== MODEL STATISTICS ==={Style.RESET_ALL}")
    print(f"Architecture:   HOPE (Nested Learning)")
    print(f"Parameters:     {Fore.YELLOW}{total_params / 1_000_000:.2f} Million{Style.RESET_ALL}")

    size_on_disk = (total_params * 4) / 1024 / 1024
    print(f"Est. File Size: {size_on_disk:.1f} MB")
    print(f"Active Memory:  {Fore.CYAN}{current_ram:.1f} MB{Style.RESET_ALL} (Unified/System RAM)")

    if DEVICE == "cuda":
        vram = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU VRAM:       {vram:.1f} MB")

    print("=" * 30 + "\n")


def load_model(path):
    print(f"{Fore.YELLOW}Loading model from {path}...{Style.RESET_ALL}")
    try:
        model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers'])
        checkpoint = torch.load(path, map_location=DEVICE)

        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            print(f"{Fore.CYAN}Detected Smart Checkpoint (step {checkpoint.get('step', 'unknown')}). Unpacking...{Style.RESET_ALL}")
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        return model

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


def generate_response(model, prompt, max_new_tokens=250, temperature=0.7, top_p=0.9, show_reasoning=True):
    # Format prompt to match training template
    full_prompt = f"Question: {prompt.strip()}\nAnswer: "

    input_ids = list(full_prompt.encode('utf-8'))
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    print(f"\n{Fore.CYAN}HOPE: {Style.RESET_ALL}", end="", flush=True)

    generated_bytes = []
    byte_buffer = bytearray()

    with torch.no_grad():
        logits, state = model(x)

    last_token_logits = logits[:, -1, :] / max(0.01, temperature)

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for batch_idx in range(last_token_logits.shape[0]):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            last_token_logits[batch_idx, indices_to_remove] = -float('Inf')

    probs = F.softmax(last_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    reasoning_started = False

    for _ in range(max_new_tokens):
        token_int = next_token.item()

        if token_int == 0:
            break

        generated_bytes.append(token_int)

        # Smart decoding & printing
        if token_int == 32:  # Space
            byte_buffer.clear()
            sys.stdout.write(" ")
        elif token_int == 10:  # Newline
            byte_buffer.clear()
            sys.stdout.write("\n")
            # Detect reasoning section start
            if show_reasoning and not reasoning_started:
                reasoning_started = True
        else:
            byte_buffer.append(int(token_int))
            try:
                decoded_char = byte_buffer.decode('utf-8')
                sys.stdout.write(decoded_char)
                byte_buffer.clear()
            except UnicodeDecodeError:
                pass

        sys.stdout.flush()

        with torch.no_grad():
            x = next_token
            logits, state = model(x, state=state)

        last_token_logits = logits[:, -1, :] / max(0.01, temperature)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for batch_idx in range(last_token_logits.shape[0]):
                indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                last_token_logits[batch_idx, indices_to_remove] = -float('Inf')

        probs = F.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    print()
    return bytes(generated_bytes).decode('utf-8', errors='ignore')


def main():
    model_path = CONFIG['save_path']

    # Also check for best checkpoint
    best_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_path) and not os.path.exists(model_path):
        model_path = best_path
        print(f"{Fore.CYAN}Using best checkpoint: {best_path}{Style.RESET_ALL}")

    model = load_model(model_path)
    print_model_stats(model)

    print("Interactive Console - Type 'quit' to exit")
    print("Commands: /temp <value>, /tokens <value>, /reasoning <on|off>")
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

            # Commands
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

            generate_response(model, user_input, max_new_tokens=max_tokens, temperature=temperature, show_reasoning=show_reasoning)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
