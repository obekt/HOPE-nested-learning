import torch
import torch.nn.functional as F
import sys
import os
import psutil  # <--- NEW: For memory tracking
import time
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
    
    # 1. Count Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Measure Memory
    # Force a garbage collection/cache clear to get accurate reading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    current_ram = get_memory_usage()
    
    print(f"\n{Fore.GREEN}=== MODEL STATISTICS ==={Style.RESET_ALL}")
    print(f"Architecture:   HOPE (Nested Learning)")
    print(f"Parameters:     {Fore.YELLOW}{total_params / 1_000_000:.2f} Million{Style.RESET_ALL}")
    
    # Estimate size on disk (Float32 = 4 bytes per param)
    size_on_disk = (total_params * 4) / 1024 / 1024
    print(f"Est. File Size: {size_on_disk:.1f} MB")
    
    # Actual RAM usage of the script
    print(f"Active Memory:  {Fore.CYAN}{current_ram:.1f} MB{Style.RESET_ALL} (Unified/System RAM)")
    
    if DEVICE == "cuda":
        vram = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU VRAM:       {vram:.1f} MB")
        
    print("="*30 + "\n")

def load_model(path):
    print(f"{Fore.YELLOW}Loading model from {path}...{Style.RESET_ALL}")
    try:
        # 1. Create the empty architecture
        model = HOPE(CONFIG['granite_model'], CONFIG['d_model'], CONFIG['n_layers'])
        
        # 2. Load the file from disk (if exists, else it will just be the backbone)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            
            # 3. INTELLIGENT UNPACKING
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                print(f"{Fore.CYAN}Detected Smart Checkpoint (includes training step). Unpacking...{Style.RESET_ALL}")
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint
                
            # 4. Inject weights into model
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"{Fore.CYAN}No checkpoint found at {path}. Using raw Granite backbone + uninitialized HOPE layers.{Style.RESET_ALL}")        
        
        # --- OPTIMIZATION: Use Half Precision (FP16) for Mac/MPS and CUDA ---
        if DEVICE in ["mps", "cuda"]:
            print(f"{Fore.YELLOW}Optimizing for GPU (Half Precision)...{Style.RESET_ALL}")
            model = model.half()
            
        model.to(DEVICE)
        model.eval()
        return model
        
    except FileNotFoundError:
        print(f"{Fore.RED}Error: Model file '{path}' not found.{Style.RESET_ALL}")
        print(f"Current Config points to: {CONFIG['save_path']}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"{Fore.RED}Architecture Mismatch!{Style.RESET_ALL}")
        print("Your saved model has different dimensions than your current CONFIG.")
        print(f"Error details: {e}")
        sys.exit(1)

def generate_response(model, prompt, max_new_tokens=150, temperature=0.7):
    # 1. Encode Prompt using Granite tokenizer
    tokenizer = model.granite.tokenizer
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    print(f"\n{Fore.CYAN}HOPE: {Style.RESET_ALL}", end="")
    
    generated_ids = []
    start_time = time.time()
    
    for _ in range(max_new_tokens):
        # Prepare tokens_dict for Granite-based forward pass
        tokens_dict = {
            'input_ids': x,
            'attention_mask': torch.ones_like(x).to(DEVICE)
        }
        
        with torch.inference_mode():
            logits = model(tokens_dict)
        
        last_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, next_token), dim=1)
        token_int = next_token.item()
        
        if token_int == tokenizer.eos_token_id: break 
        
        generated_ids.append(token_int)
        
        # Stream the token
        decoded_part = tokenizer.decode([token_int])
        sys.stdout.write(decoded_part)
        sys.stdout.flush()

    end_time = time.time()
    duration = end_time - start_time
    num_tokens = len(generated_ids)
    tps = num_tokens / duration if duration > 0 else 0
    
    print(f"\n\n{Fore.YELLOW}[Stats: {num_tokens} tokens | {duration:.2f}s | {tps:.1f} tok/s]{Style.RESET_ALL}")
    return tokenizer.decode(generated_ids)

def main():
    # Use the English model path
    model_path = CONFIG['save_path']  # Will use hope_en_deep.pth from CONFIG
    
    # 1. Load
    model = load_model(model_path)
    
    # 2. Show Stats (The new part)
    print_model_stats(model)
    
    print("Interactive Console - Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input(f"\n{Fore.WHITE}You: {Style.RESET_ALL}")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            if not user_input.strip():
                continue
                
            generate_response(model, user_input)
            
            # Optional: Show memory spike after generation
            # print(f"{Fore.BLACK}{Style.DIM}[Mem: {get_memory_usage():.0f}MB]{Style.RESET_ALL}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
