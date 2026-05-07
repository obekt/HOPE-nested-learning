import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import time
import sys
import os
import json
from datetime import timedelta
from colorama import Fore, Style, init

# Initialize color output
init(autoreset=True)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "d_model": 512,           # Width (Reasoning capability)
    "n_layers": 16,           # Depth
    "vocab_size": 256,        # Byte-Level
    "seq_len": 512,           # Training window

    # Training Scale
    "batch_size": 4,
    "accumulate_grad": 8,     # Effective batch of 32

    # Learning
    "learning_rate": 1e-3,    # Slightly higher for faster convergence on smaller data
    "max_steps": 3000,        # ~1.2 epochs on 77k rows with effective batch 32
    "warmup_steps": 500,
    "weight_decay": 0.05,
    "grad_clip": 1.0,

    # --- DATASET SETTINGS ---
    "dataset_name": "obekt/obekt-question-answer-reasoning-micro-v0.1",
    "dataset_config": None,
    "dataset_columns": "question, answer, reasoning",
    "max_samples": 80000,
    "isolate_samples": True,  # ISOLATED mode for Q&A data

    # --- CHECKPOINTING ---
    "save_path": "hope_qa_micro.pth",
    "checkpoint_every": 500,  # Save every N steps
    "log_file": "training.log",
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
        self.decay_param = nn.Parameter(torch.tensor(0.0))  # Logits for sigmoid

    def forward(self, x, state=None, mask=None):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        k = F.elu(k) + 1.0
        batch_size, seq_len, _ = x.shape

        # Use provided state or initialize new memory
        memory = state if state is not None else torch.zeros(batch_size, self.dim, self.dim, device=x.device, dtype=x.dtype)

        if seq_len == 0:
            return torch.zeros_like(x), memory

        outputs = []
        for t in range(seq_len):
            q_t = q[:, t, :].unsqueeze(1)
            k_t = k[:, t, :].unsqueeze(1)
            v_t = v[:, t, :].unsqueeze(1)
            read_out = torch.bmm(q_t, memory)
            update = torch.bmm(k_t.transpose(1, 2), v_t)
            decay_value = torch.sigmoid(self.decay_param)

            # --- MASKED UPDATE FIX ---
            if mask is not None:
                m_t = mask[:, t].view(batch_size, 1, 1)
                memory = (1 - m_t) * memory + m_t * (decay_value * memory + update)
            else:
                memory = decay_value * memory + update

            outputs.append(read_out)

        out = torch.cat(outputs, dim=1)
        return self.proj_out(out), memory


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
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fast_memory = SelfModifyingLayer(d_model)
        self.norm_fast = nn.LayerNorm(d_model)
        self.cms_layers = nn.ModuleList([
            ContinuumMemoryBlock(d_model) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        # Better initialization for training stability
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, state=None):
        # Create Padding Mask (1 for real tokens, 0 for padding 0)
        mask = (x != 0).float()

        h = self.embedding(x)
        # Pass mask to ensure memory doesn't leak during padding
        fast_out, new_state = self.fast_memory(h, state=state, mask=mask)
        h = self.norm_fast(h + fast_out)
        for layer in self.cms_layers:
            h = layer(h)
        return self.head(h), new_state


# ==========================================
# 3. ROBUST DATASET
# ==========================================

def format_qa_item(item, columns):
    """Format Q&A datasets with proper templates."""
    parts = []
    col_set = set(c.strip().lower() for c in columns)

    # Specific formatting for question/answer/reasoning
    if "question" in col_set and "answer" in col_set:
        q = item.get("question", item.get("Question", ""))
        a = item.get("answer", item.get("Answer", ""))
        r = item.get("reasoning", item.get("Reasoning", ""))

        if q and a:
            parts.append(f"Question: {q.strip()}")
            parts.append(f"Answer: {a.strip()}")
            if r:
                parts.append(f"Reasoning: {r.strip()}")
            return "\n".join(parts) + "\n"

    # Generic multi-column formatting
    for col in columns:
        col = col.strip()
        val = item.get(col)
        if val and isinstance(val, str) and len(val.strip()) > 0:
            parts.append(f"{col.capitalize()}: {val.strip()}")

    if parts:
        return "\n".join(parts) + "\n"
    return ""


class SmartTextDataset(IterableDataset):
    def __init__(self, dataset_name, dataset_config, seq_len, target_columns=None, max_samples=50000, split="train"):
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.detected_columns = []
        self.split = split
        self.samples_yielded = 0

        # Parse the user's column preference
        self.target_columns = None
        if target_columns and isinstance(target_columns, str):
            self.target_columns = [c.strip() for c in target_columns.split(',') if c.strip()]
            print(f"{Fore.YELLOW}Forcing columns: {self.target_columns}{Style.RESET_ALL}")

        print(f"{Fore.YELLOW}Connecting to Hugging Face: {dataset_name} (split={split})...{Style.RESET_ALL}")
        if dataset_config:
            self.hf_dataset = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
        else:
            self.hf_dataset = load_dataset(dataset_name, split=split, streaming=True)

    def _process_item(self, item):
        # Update dashboard info
        if not self.detected_columns:
            self.detected_columns = list(item.keys())

        # --- MODE 1: USER SPECIFIED COLUMNS ---
        if self.target_columns:
            text = format_qa_item(item, self.target_columns)
            if text:
                return text
            # Fallback to generic
            text_parts = []
            for col in self.target_columns:
                val = item.get(col)
                if val and isinstance(val, str) and len(val.strip()) > 0:
                    text_parts.append(val.strip())
            if text_parts:
                return "\n".join(text_parts) + "\n"
            return ""

        # --- MODE 2: AUTO-DETECT (Universal) ---
        if 'text' in item and 'title' in item:
            return f"{item['title']}\n{item['text']}\n"

        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 20:
                text_parts.append(value)

        return "\n".join(text_parts) + "\n"

    def __iter__(self):
        iterator = iter(self.hf_dataset)
        count = 0
        buffer = []

        # Check if we should isolate rows (Good for Q&A) or pack them (Good for Wikipedia)
        isolate = CONFIG.get('isolate_samples', False)

        while count < self.max_samples:
            try:
                item = next(iterator)
                text = self._process_item(item)
                if not text:
                    continue

                tokens = list(text.encode('utf-8'))

                if isolate:
                    # --- ISOLATED MODE: Each row is its own training example ---
                    tokens = (tokens[:self.seq_len]) + [0]
                    padding_needed = (self.seq_len + 1) - len(tokens)
                    if padding_needed > 0:
                        tokens.extend([0] * padding_needed)

                    yield torch.tensor(tokens, dtype=torch.long)
                    count += 1
                else:
                    # --- PACKED MODE: Stitch multiple rows together ---
                    if buffer and buffer[-1] != 10:
                        buffer.append(10)
                    buffer.extend(tokens)

                    while len(buffer) >= self.seq_len + 1:
                        yield torch.tensor(buffer[:self.seq_len + 1], dtype=torch.long)
                        buffer = buffer[self.seq_len:]
                        count += 1
                        if count >= self.max_samples:
                            break

            except StopIteration:
                break
            except Exception as e:
                print(f"{Fore.RED}Data error: {e}{Style.RESET_ALL}")
                continue


# ==========================================
# 4. TRAINING WITH DASHBOARD
# ==========================================

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def decode_preview(tensor):
    try:
        tokens = tensor[0].tolist()[:120]
        text = bytes(tokens).decode('utf-8', errors='ignore')
        return text.replace('\n', ' ')
    except:
        return "..."


def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')


def draw_dashboard(step, max_steps, loss, val_loss, speed, eta, columns, preview_text, lr):
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

    val_str = f" | Val: {Fore.MAGENTA}{val_loss:.4f}{Style.RESET_ALL}" if val_loss is not None else ""
    print(f"Progress: [{Fore.GREEN}{bar}{Style.RESET_ALL}] {step}/{max_steps}")
    print(f"Stats:    Loss: {Fore.RED}{loss:.4f}{Style.RESET_ALL}{val_str} | Speed: {speed:.0f} tok/s | ETA: {eta}")
    print(f"LR:       {lr:.2e}")
    print("-" * 60)

    print(f"{Style.DIM}Press Ctrl+C to stop and save.{Style.RESET_ALL}")


def log_to_file(msg):
    try:
        with open(CONFIG.get('log_file', 'training.log'), 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass


def run_validation(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, CONFIG['vocab_size']), targets.reshape(-1), ignore_index=0)
            total_loss += loss.item()
            count += 1
            if count >= 50:  # Limit validation to 50 batches for speed
                break
    model.train()
    return total_loss / max(count, 1)


def train():
    # Setup
    model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers']).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG.get('weight_decay', 0.01),
        betas=(0.9, 0.95)
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
    best_val_loss = float('inf')
    if os.path.exists(CONFIG['save_path']):
        checkpoint = torch.load(CONFIG['save_path'], map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            start_step = checkpoint.get('step', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            print(f"{Fore.GREEN}Resumed from step {start_step}, best val loss: {best_val_loss:.4f}{Style.RESET_ALL}")
        else:
            model.load_state_dict(checkpoint)

    # Data - Train
    train_dataset = SmartTextDataset(
        CONFIG['dataset_name'],
        CONFIG['dataset_config'],
        CONFIG['seq_len'],
        target_columns=CONFIG.get('dataset_columns'),
        max_samples=CONFIG.get('max_samples', 50000),
        split="train"
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'])

    # Data - Validation (use a small subset of train for now if no val split)
    val_dataset = SmartTextDataset(
        CONFIG['dataset_name'],
        CONFIG['dataset_config'],
        CONFIG['seq_len'],
        target_columns=CONFIG.get('dataset_columns'),
        max_samples=1000,
        split="train"
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    model.train()
    iter_loader = iter(train_loader)
    step = start_step
    running_loss = 0
    val_loss = None

    start_time = time.time()
    last_update_time = time.time()
    last_val_time = time.time()

    current_preview = "Waiting for data..."

    try:
        while step < CONFIG['max_steps']:
            t0 = time.time()
            optimizer.zero_grad()

            for _ in range(CONFIG['accumulate_grad']):
                try:
                    batch = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(train_loader)
                    batch = next(iter_loader)

                inputs = batch[:, :-1].to(DEVICE)
                targets = batch[:, 1:].to(DEVICE)

                # Capture text for dashboard
                current_preview = decode_preview(inputs)

                if scaler:
                    with torch.cuda.amp.autocast():
                        logits, _ = model(inputs)
                        loss = F.cross_entropy(logits.reshape(-1, CONFIG['vocab_size']), targets.reshape(-1), ignore_index=0)
                    scaler.scale(loss).backward()
                else:
                    logits, _ = model(inputs)
                    loss = F.cross_entropy(logits.reshape(-1, CONFIG['vocab_size']), targets.reshape(-1), ignore_index=0)
                    loss.backward()

                running_loss += loss.item()

            # Gradient clipping
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.get('grad_clip', 1.0))

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            step += 1

            # --- VALIDATION ---
            if time.time() - last_val_time > 60:  # Every 60 seconds
                val_loss = run_validation(model, val_loader, DEVICE)
                last_val_time = time.time()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save best model
                    best_path = CONFIG['save_path'].replace('.pth', '_best.pth')
                    torch.save({
                        'model_state': model.state_dict(),
                        'step': step,
                        'best_val_loss': best_val_loss,
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict()
                    }, best_path)
                    print(f"\n{Fore.GREEN}New best model saved! Val loss: {best_val_loss:.4f}{Style.RESET_ALL}")

            # --- PERIODIC CHECKPOINT ---
            if step % CONFIG.get('checkpoint_every', 500) == 0:
                checkpoint_data = {
                    'model_state': model.state_dict(),
                    'step': step,
                    'best_val_loss': best_val_loss,
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict()
                }
                torch.save(checkpoint_data, CONFIG['save_path'])
                print(f"\n{Fore.CYAN}Checkpoint saved at step {step}{Style.RESET_ALL}")

            # --- UPDATE DASHBOARD ---
            if time.time() - last_update_time > 0.2:
                dt = time.time() - t0
                dt = max(dt, 0.001)

                tokens_per_sec = (CONFIG['batch_size'] * CONFIG['seq_len'] * CONFIG['accumulate_grad']) / dt
                avg_loss = running_loss / CONFIG['accumulate_grad']
                running_loss = 0
                eta_seconds = (CONFIG['max_steps'] - step) * dt
                current_lr = scheduler.get_last_lr()[0]

                draw_dashboard(
                    step,
                    CONFIG['max_steps'],
                    avg_loss,
                    val_loss,
                    tokens_per_sec,
                    format_time(eta_seconds),
                    train_dataset.detected_columns,
                    current_preview,
                    current_lr
                )
                val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                log_msg = f"Step {step}/{CONFIG['max_steps']} | Loss: {avg_loss:.4f} | Val: {val_str} | LR: {current_lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s"
                log_to_file(log_msg)
                last_update_time = time.time()

    except KeyboardInterrupt:
        pass  # Handle save below

    print(f"\n{Fore.GREEN}Saving to {CONFIG['save_path']}...{Style.RESET_ALL}")
    checkpoint_data = {
        'model_state': model.state_dict(),
        'step': step,
        'best_val_loss': best_val_loss,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }
    torch.save(checkpoint_data, CONFIG['save_path'])
    print("Done.")


if __name__ == "__main__":
    train()
