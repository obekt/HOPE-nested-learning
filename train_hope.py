import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import time
import sys
import os
from datetime import timedelta
from colorama import Fore, Style, init
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Initialize color output
init(autoreset=True)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "granite_model": "ibm-granite/granite-embedding-small-english-r2",
    "d_model": 384,           # Dimension of Granite-small
    "n_layers": 32,           # Fast Memory layers
    "seq_len": 512,           # Adjusted for Granite transformer
    
    # Speed Adjustments
    "batch_size": 4,          
    "accumulate_grad": 8,     
    
    # Optimized Learning Rate
    "learning_rate": 2e-4,    
    "max_steps": 10000,       
    
    # Warmup and scheduler
    "warmup_steps": 500,      
    "weight_decay": 0.01,     

    # --- DATASET SETTINGS ---
    "dataset_name": "wikimedia/wikipedia",
    "dataset_config": "20231101.en",  
    "dataset_columns": "title, text",
    "max_samples": 50000,    

    "save_path": "hope_granite_hybrid.pth"
}

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ==========================================
# 2. NESTED LEARNING ARCHITECTURE (HOPE)
# ==========================================

class SelfModifyingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.decay_param = nn.Parameter(torch.tensor(0.0)) # Logits for sigmoid

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        k = F.elu(k) + 1.0 
        decay_value = torch.sigmoid(self.decay_param)
        
        # JIT-compiled recurrent loop for stability and speed
        # This replaces the numerically unstable power-based vectorization
        out = self._recurrent_forward(q, k, v, decay_value)
        
        return self.proj_out(out)

    @torch.jit.export
    def _recurrent_forward(self, q, k, v, decay):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
        batch_size, seq_len, dim = q.shape
        memory = torch.zeros(batch_size, dim, dim, device=q.device, dtype=q.dtype)
        outputs = torch.zeros_like(q)
        
        for t in range(seq_len):
            q_t = q[:, t, :].unsqueeze(1) # [B, 1, D]
            k_t = k[:, t, :].unsqueeze(1) # [B, 1, D]
            v_t = v[:, t, :].unsqueeze(1) # [B, 1, D]
            
            # Read out from memory
            read_out = torch.bmm(q_t, memory) # [B, 1, D]
            outputs[:, t, :] = read_out.squeeze(1)
            
            # Update memory
            update = torch.bmm(k_t.transpose(1, 2), v_t) # [B, D, D]
            memory = decay * memory + update
            
        return outputs

class ContinuumMemoryBlock(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class HOPE(nn.Module):
    def __init__(self, granite_model_name, d_model, n_layers):
        super().__init__()
        print(f"Loading Granite Backbone: {granite_model_name}...")
        self.granite = SentenceTransformer(granite_model_name)
        
        # Freeze Granite weights (Slow Memory)
        for param in self.granite.parameters():
            param.requires_grad = False
            
        self.fast_memory = SelfModifyingLayer(d_model)
        self.norm_fast = nn.LayerNorm(d_model)
        self.cms_layers = nn.ModuleList([
            ContinuumMemoryBlock(d_model) for _ in range(n_layers)
        ])
        
        # Prediction head: predicts next token in Granite's vocab
        vocab_size = self.granite.tokenizer.vocab_size
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens_dict):
        # 1. Get static embeddings from IBM Granite (Slow Memory)
        with torch.no_grad():
            outputs = self.granite[0](tokens_dict)
            slow_embeddings = outputs['token_embeddings'] # (Batch, Seq, 384)
            
        # 2. Pass through Fast Memory (Real-time adaptation)
        fast_out = self.fast_memory(slow_embeddings)
        h = self.norm_fast(slow_embeddings + fast_out)
        
        # 3. Continuum Memory Layers
        for layer in self.cms_layers:
            h = layer(h)
            
        return self.head(h)

# ==========================================
# 3. ROBUST DATASET (Strict Filter + Dashboard Fix)
# ==========================================

class SmartTextDataset(IterableDataset):
    def __init__(self, dataset_name, dataset_config, seq_len, target_columns=None, max_samples=50000):
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.detected_columns = []
        
        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['granite_model'], model_max_length=100000)
        
        # Parse the user's column preference
        self.target_columns = None
        if target_columns and isinstance(target_columns, str):
            # Convert "title, text" -> ['title', 'text']
            self.target_columns = [c.strip() for c in target_columns.split(',') if c.strip()]
            print(f"{Fore.YELLOW}Forcing columns: {self.target_columns}{Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}Connecting to Hugging Face Stream: {dataset_name}...{Style.RESET_ALL}")
        if dataset_config:
            self.hf_dataset = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
        else:
            self.hf_dataset = load_dataset(dataset_name, split="train", streaming=True)

    def _process_item(self, item):
        # Update dashboard info
        if not self.detected_columns:
            self.detected_columns = list(item.keys())

        # --- MODE 1: USER SPECIFIED COLUMNS ---
        if self.target_columns:
            text_parts = []
            for col in self.target_columns:
                # Get the value if the column exists
                val = item.get(col)
                if val and isinstance(val, str) and len(val.strip()) > 0:
                    text_parts.append(val.strip())
            
            # If we found data in the requested columns, return it
            if text_parts:
                return "\n".join(text_parts)
            return "" # Skip row if requested columns are empty

        # --- MODE 2: AUTO-DETECT (Universal) ---
        # If config is None, we grab everything that looks like text
        
        # Special Case: Wikipedia (Auto-detect Title/Text if not specified)
        if 'text' in item and 'title' in item:
            return f"{item['title']}\n{item['text']}"
            
        # Fallback: Grab all string columns
        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 20: # Filter short IDs
                text_parts.append(value)
        
        return "\n".join(text_parts)

    def __iter__(self):
        iterator = iter(self.hf_dataset)
        count = 0
        while count < self.max_samples:
            try:
                item = next(iterator)
                text = self._process_item(item)
                
                if not text or len(text) < 100: 
                    continue 
                
                # Use Granite tokenizer with truncation to avoid warnings on huge docs
                tokens = self.tokenizer.encode(
                    text, 
                    add_special_tokens=False, 
                    truncation=True, 
                    max_length=8192 # Respect model's absolute limit
                )
                
                for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                    chunk = tokens[i : i + self.seq_len + 1]
                    if len(chunk) == self.seq_len + 1:
                        yield torch.tensor(chunk, dtype=torch.long)
                        count += 1
                        if count >= self.max_samples: break
            except StopIteration:
                break
            except Exception:
                continue

# ==========================================
# 4. TRAINING WITH DASHBOARD
# ==========================================

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def decode_preview(tensor, tokenizer):
    try:
        # Decode first item in batch, first 20 tokens
        tokens = tensor[0].tolist()[:20] 
        text = tokenizer.decode(tokens)
        return text.replace('\n', ' ') 
    except:
        return "..."

def clear_screen():
    # Cross-platform clear
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_dashboard(step, max_steps, loss, speed, eta, columns, preview_text):
    clear_screen()
    
    # 1. Header
    print(f"{Fore.GREEN}=== HOPE NESTED LEARNING DASHBOARD ==={Style.RESET_ALL}")
    print(f"Device: {Fore.CYAN}{DEVICE}{Style.RESET_ALL} | Model: {CONFIG['d_model']} dim / {CONFIG['n_layers']} layers")
    print(f"Dataset: {CONFIG['dataset_name']}")
    
    # 2. Columns Found
    col_str = ", ".join(columns) if columns else "Scanning..."
    print(f"Columns Found: {Fore.YELLOW}[ {col_str} ]{Style.RESET_ALL}")
    print("-" * 60)
    
    # 3. Live Data Preview
    print(f"{Fore.BLUE}Live Input Data:{Style.RESET_ALL}")
    print(f"\"{preview_text}...\"")
    print("-" * 60)
    
    # 4. Progress Bar
    bar_len = 30
    filled_len = int(bar_len * step // max_steps)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    
    print(f"Progress: [{Fore.GREEN}{bar}{Style.RESET_ALL}] {step}/{max_steps}")
    print(f"Stats:    Loss: {Fore.RED}{loss:.4f}{Style.RESET_ALL} | Speed: {speed:.0f} tok/s | ETA: {eta}")
    print("-" * 60)
    
    # FIX IS HERE: Use Style.DIM instead of Fore.DIM
    print(f"{Style.DIM}Press Ctrl+C to stop and save.{Style.RESET_ALL}")

def train():
    # Setup
    model = HOPE(CONFIG['granite_model'], CONFIG['d_model'], CONFIG['n_layers']).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG.get('weight_decay', 0.01)
    )
    
    # Learning rate scheduler with warmup
    def get_lr(step):
        warmup_steps = CONFIG.get('warmup_steps', 500)
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (CONFIG['max_steps'] - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    
    # Resume Logic
    start_step = 0
    if os.path.exists(CONFIG['save_path']):
        checkpoint = torch.load(CONFIG['save_path'], map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            start_step = checkpoint['step']
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
        else:
            model.load_state_dict(checkpoint)

    # Data
    dataset = SmartTextDataset(
        CONFIG['dataset_name'],
        CONFIG['dataset_config'],
        CONFIG['seq_len'],
        target_columns=CONFIG.get('dataset_columns'),
        max_samples=CONFIG.get('max_samples', 50000)
    )
    
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'])
    
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    
    model.train()
    iter_loader = iter(loader)
    step = start_step
    running_loss = 0
    
    start_time = time.time()
    last_update_time = time.time()
    
    current_preview = "Waiting for data..."
    
    try:
        while step < CONFIG['max_steps']:
            t0 = time.time()
            optimizer.zero_grad()
            
            for _ in range(CONFIG['accumulate_grad']):
                try:
                    batch = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(loader) 
                    batch = next(iter_loader)
                
                inputs_ids = batch[:, :-1].to(DEVICE)
                targets = batch[:, 1:].to(DEVICE)
                
                # Construct tokens_dict for Granite
                # We reuse input_ids as token_embeddings input
                tokens_dict = {
                    'input_ids': inputs_ids,
                    'attention_mask': torch.ones_like(inputs_ids).to(DEVICE)
                }
                
                # Capture text for dashboard
                current_preview = decode_preview(inputs_ids, model.granite.tokenizer)
                
                # Mixed precision context
                autocast_device = "cuda" if DEVICE == "cuda" else "mps" if DEVICE == "mps" else None
                
                if autocast_device:
                    with torch.autocast(device_type=autocast_device):
                        logits = model(tokens_dict)
                        loss = F.cross_entropy(logits.reshape(-1, model.head.out_features), targets.reshape(-1))
                    
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    logits = model(tokens_dict)
                    loss = F.cross_entropy(logits.reshape(-1, model.head.out_features), targets.reshape(-1))
                    loss.backward()
                
                running_loss += loss.item()

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()  # Update learning rate
            step += 1
            
            # --- UPDATE DASHBOARD (Every 0.2s to prevent flicker) ---
            if time.time() - last_update_time > 0.2:
                dt = time.time() - t0
                # Avoid division by zero on first step
                dt = max(dt, 0.001) 
                
                tokens_per_sec = (CONFIG['batch_size'] * CONFIG['seq_len'] * CONFIG['accumulate_grad']) / dt
                avg_loss = running_loss / CONFIG['accumulate_grad']
                running_loss = 0
                eta_seconds = (CONFIG['max_steps'] - step) * dt
                
                draw_dashboard(
                    step, 
                    CONFIG['max_steps'], 
                    avg_loss, 
                    tokens_per_sec, 
                    format_time(eta_seconds),
                    dataset.detected_columns, # Pass detected columns
                    current_preview
                )
                last_update_time = time.time()

    except KeyboardInterrupt:
        pass # Handle save below
    
    print(f"\n{Fore.GREEN}Saving to {CONFIG['save_path']}...{Style.RESET_ALL}")
    checkpoint_data = {
        'model_state': model.state_dict(),
        'step': step,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }
    torch.save(checkpoint_data, CONFIG['save_path'])
    print("Done.")

if __name__ == "__main__":
    train()
