import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import time
import sys
import os
import json
import queue
import threading
from datetime import timedelta
from colorama import Fore, Style, init

# Initialize color output
init(autoreset=True)

# ==========================================
# 0. TOKENIZER
# ==========================================
print(f"{Fore.YELLOW}Loading GPT-2 tokenizer...{Style.RESET_ALL}")
TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")  # Rust tokenizer, ~3x faster; identical vocab/IDs
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.model_max_length = 1_000_000_000  # Suppress max_length warnings on long articles
PAD_TOKEN_ID = TOKENIZER.pad_token_id
EOS_TOKEN_ID = TOKENIZER.eos_token_id
VOCAB_SIZE = TOKENIZER.vocab_size
print(f"{Fore.GREEN}Tokenizer ready: vocab_size={VOCAB_SIZE}, pad_token_id={PAD_TOKEN_ID}{Style.RESET_ALL}")

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "d_model": 512,
    "n_layers": 16,
    "vocab_size": VOCAB_SIZE,
    "seq_len": 512,

    # Nested learning: CMS tiers as [n_layers, update_period] pairs.
    # Layer counts must sum to n_layers. Period = optimizer steps between
    # updates for that tier (gradients are averaged in between).
    # Fast tier also owns embedding, fast_memory, norms and head.
    "cms_tiers": [[8, 1], [5, 4], [3, 16]],

    "batch_size": 4,
    "accumulate_grad": 8,

    # Fast-memory scan: exact chunk-parallel evaluation of the delta rule for
    # sequences longer than 1 token (training + prefill). Token-by-token loop
    # is kept for single-token generation. fast_force_loop=True disables the
    # chunked path (debug/benchmarking). Runtime-only; not part of checkpoints.
    "fast_chunk_size": 64,
    "fast_force_loop": False,

    "learning_rate": 2e-4,
    "max_steps": 12000,
    "warmup_steps": 1500,
    "weight_decay": 0.05,
    "grad_clip": 1.0,

    "dataset_name": "wikimedia/wikipedia",
    "dataset_config": "20231101.en",
    "dataset_columns": "title, text",
    "max_samples": 500000,
    "isolate_samples": False,
    "train_skip_samples": 0,       # articles to skip at stream start (used by the 48k extension, see EXPERIMENT_LOG D4)
    "val_skip_samples": 100000,    # val split offset in articles; must stay beyond training coverage

    "save_path": "hope_foundation.pth",
    "checkpoint_every": 1000,
    "log_file": "foundation.log",
}

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ==========================================
# 2. NESTED LEARNING ARCHITECTURE (HOPE)
# ==========================================

_TRI_SOLVE_OK_CACHE = {}

def _tri_solve_ok(device, dtype=torch.float32):
    """One-time probe: does this device support triangular solves?
    Falls back to the per-token loop where it doesn't."""
    key = (str(device), dtype)
    if key not in _TRI_SOLVE_OK_CACHE:
        try:
            a = torch.eye(2, device=device, dtype=dtype)
            b = torch.ones(2, 1, device=device, dtype=dtype)
            torch.linalg.solve_triangular(a, b, upper=False, unitriangular=True)
            _TRI_SOLVE_OK_CACHE[key] = True
        except Exception:
            _TRI_SOLVE_OK_CACHE[key] = False
    return _TRI_SOLVE_OK_CACHE[key]


def _chunked_delta_scan(q, k, v, log_alpha, beta, memory, mask, chunk_size):
    """Exact chunk-parallel evaluation of the gated delta-rule recurrence

        M_t = alpha_t * M_{t-1} + beta_t * k_t^T (v_t - k_t M_{t-1})
        o_t = q_t M_{t-1}   (read-out BEFORE the write at t)

    Mathematically identical to the per-token loop (verified to ~1e-6 fp32),
    but the T sequential small ops become T/chunk_size batched matmul steps —
    ~27x faster fwd+bwd on MPS at B=4, T=512, D=512.

    Derivation: with G_t = cumsum(log alpha), all decay factors appear as
    pairwise ratios exp(G_t - G_s) <= 1 (no division by tiny cumprods). The
    per-token write vectors E_t = beta_t (v_t - k_t M_{t-1}) solve the unit
    lower-triangular system (I + A) E = V - (K e^{G_{t-1}}) M_0 within each
    chunk, where A_ts = exp(G_{t-1} - G_s) beta_s (k_t . k_s) for s < t.

    Shapes: q, k, v [B, T, D]; log_alpha, beta [B, T, 1]; memory [B, D, D];
    mask [B, T] float (1 = real token) or None. Returns (out [B, T, D] before
    proj_out, final memory).
    """
    batch_size, seq_len, _ = q.shape
    al = log_alpha.squeeze(-1)   # [B, T] log-alpha
    be = beta.squeeze(-1)        # [B, T] beta
    if mask is not None:
        # Fold the mask into the gates: masked token -> alpha=1, beta=0,
        # i.e. memory passes through untouched while the read-out is still
        # produced — exactly what the loop path's mask branch does.
        al = al * mask
        be = be * mask

    outs = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)   # partial final chunk at true length
        qc, kc, vc = q[:, start:end], k[:, start:end], v[:, start:end]
        alc, bec = al[:, start:end], be[:, start:end]
        length = end - start

        g = torch.cumsum(alc, dim=1)             # [B, L]   G_t
        gs = g - alc                             # [B, L]   G_{t-1}
        # decay[t, s] = exp(G_{t-1} - G_s) <= 1  (exponents are always <= 0)
        decay = torch.exp(gs.unsqueeze(2) - g.unsqueeze(1))            # [B, L, L]
        kkT = torch.bmm(kc, kc.transpose(1, 2))                        # [B, L, L]
        A = torch.tril(decay * kkT * bec.unsqueeze(1), diagonal=-1)    # beta_s on columns
        eye = torch.eye(length, device=A.device, dtype=A.dtype)

        # (I + A) E = V - (K e^{G_{t-1}}) M_0
        rhs = vc - torch.bmm(kc * torch.exp(gs).unsqueeze(-1), memory)
        E = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)

        # o_t = e^{G_{t-1}} (q_t M_0) + sum_{s<t} exp(G_{t-1}-G_s) (q_t.k_s) beta_s E_s
        qkT = torch.bmm(qc, kc.transpose(1, 2))
        o = torch.bmm(qc * torch.exp(gs).unsqueeze(-1), memory) \
            + torch.bmm(torch.tril(decay * qkT, diagonal=-1), bec.unsqueeze(-1) * E)
        outs.append(o)

        # M_next = e^{G_L} M_0 + sum_s exp(G_L - G_s) beta_s k_s^T E_s
        gL = g[:, -1].unsqueeze(1)               # [B, 1]
        wK = kc * torch.exp(gL - g).unsqueeze(-1) * bec.unsqueeze(-1)
        memory = torch.exp(gL).unsqueeze(-1) * memory + torch.bmm(wK.transpose(1, 2), E)

    return torch.cat(outs, dim=1), memory


class SelfModifyingLayer(nn.Module):
    """Fast-weight memory trained by an inner-loop delta rule.

    The memory matrix M performs one step of online gradient descent per token
    on the reconstruction loss ||k M - v||^2:

        M_t = alpha_t * M_{t-1} + beta_t * k_t^T (v_t - k_t M_{t-1})

    where alpha_t (forget gate) and beta_t (inner learning rate) are
    input-dependent, learned per token. This is genuine inner-loop learning:
    the write is proportional to the memory's prediction *error*, so content
    the memory already knows is not re-written.

    Sequences longer than one token are evaluated with an exact chunk-parallel
    form of the same recurrence (_chunked_delta_scan); single-token calls
    (incremental generation) use the per-token loop.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        # Per-token gates: alpha (forget/retain) and beta (inner LR)
        self.gate_alpha = nn.Linear(dim, 1)
        self.gate_beta = nn.Linear(dim, 1)

    def _forward_loop(self, q, k, v, alpha, beta, memory, mask):
        """Per-token recurrence — kept verbatim for single-token generation."""
        batch_size, seq_len, _ = q.shape
        outputs = []
        for t in range(seq_len):
            q_t = q[:, t, :].unsqueeze(1)   # [B, 1, D]
            k_t = k[:, t, :].unsqueeze(1)   # [B, 1, D]
            v_t = v[:, t, :].unsqueeze(1)   # [B, 1, D]
            a_t = alpha[:, t, :].unsqueeze(1)  # [B, 1, 1]
            b_t = beta[:, t, :].unsqueeze(1)   # [B, 1, 1]

            read_out = torch.bmm(q_t, memory)

            # Inner-loop SGD step: write only the prediction error
            pred = torch.bmm(k_t, memory)              # [B, 1, D] what M recalls for k
            error = v_t - pred                          # [B, 1, D]
            update = b_t * torch.bmm(k_t.transpose(1, 2), error)
            new_memory = a_t * memory + update

            if mask is not None:
                m_t = mask[:, t].view(batch_size, 1, 1)
                memory = (1 - m_t) * memory + m_t * new_memory
            else:
                memory = new_memory

            outputs.append(read_out)

        return torch.cat(outputs, dim=1), memory

    def forward(self, x, state=None, mask=None):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        # Normalize keys so the delta-rule inner step is well-conditioned
        # (||k||=1 makes beta a true step size and bounds the update).
        k = F.normalize(k, dim=-1)
        batch_size, seq_len, _ = x.shape

        # alpha near 1.0 at init (retain), beta small at init (gentle writes)
        z_alpha = self.gate_alpha(x) + 4.0              # [B, T, 1] pre-activation
        beta = torch.sigmoid(self.gate_beta(x) - 2.0)   # [B, T, 1]

        memory = state if state is not None else torch.zeros(batch_size, self.dim, self.dim, device=x.device, dtype=x.dtype)

        if seq_len == 0:
            return torch.zeros_like(x), memory

        chunk_size = CONFIG.get("fast_chunk_size", 64)
        use_chunked = (
            seq_len > 1
            and chunk_size
            and not CONFIG.get("fast_force_loop", False)
            and _tri_solve_ok(x.device, x.dtype)
        )
        if use_chunked:
            # logsigmoid on the pre-activation is exactly log(sigmoid(.)) and
            # never hits log(0); exp() of it underflows to 0 like sigmoid does.
            out, memory = _chunked_delta_scan(
                q, k, v, F.logsigmoid(z_alpha), beta, memory, mask, chunk_size)
        else:
            alpha = torch.sigmoid(z_alpha)   # [B, T, 1]
            out, memory = self._forward_loop(q, k, v, alpha, beta, memory, mask)

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
    def __init__(self, vocab_size, d_model, n_layers, cms_tiers=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fast_memory = SelfModifyingLayer(d_model)
        self.norm_fast = nn.LayerNorm(d_model)
        self.cms_layers = nn.ModuleList([
            ContinuumMemoryBlock(d_model) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

        # Nested learning tiers: [[n_layers, update_period], ...]
        # Earlier CMS layers -> faster tiers, later layers -> slower tiers.
        if cms_tiers is None:
            cms_tiers = [[n_layers, 1]]
        assert sum(n for n, _ in cms_tiers) == n_layers, \
            f"cms_tiers layer counts {[n for n, _ in cms_tiers]} must sum to n_layers={n_layers}"
        self.cms_tiers = [list(t) for t in cms_tiers]

        self._init_weights()

    def tier_param_groups(self):
        """Partition parameters into (period, params) groups for nested updates.

        Tier 0 (fastest) also owns the embedding, fast memory, norm and head,
        since those must track data at the highest frequency.
        """
        groups = []
        layer_idx = 0
        for i, (n, period) in enumerate(self.cms_tiers):
            params = []
            if i == 0:
                params += list(self.embedding.parameters())
                params += list(self.fast_memory.parameters())
                params += list(self.norm_fast.parameters())
                params += list(self.head.parameters())
            for layer in self.cms_layers[layer_idx:layer_idx + n]:
                params += list(layer.parameters())
            layer_idx += n
            groups.append((period, params))
        return groups

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, state=None, mask=None):
        if mask is None:
            mask = (x != PAD_TOKEN_ID).float()
        h = self.embedding(x)
        fast_out, new_state = self.fast_memory(h, state=state, mask=mask)
        h = self.norm_fast(h + fast_out)
        for layer in self.cms_layers:
            h = layer(h)
        return self.head(h), new_state


# ==========================================
# 3. ROBUST DATASET
# ==========================================

def format_qa_item(item, columns):
    parts = []
    col_set = set(c.strip().lower() for c in columns)

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

    for col in columns:
        col = col.strip()
        val = item.get(col)
        if val and isinstance(val, str) and len(val.strip()) > 0:
            parts.append(f"{col.capitalize()}: {val.strip()}")

    if parts:
        return "\n".join(parts) + "\n"
    return ""


class SmartTextDataset(IterableDataset):
    def __init__(self, dataset_name, dataset_config, seq_len, target_columns=None, max_samples=50000, split="train", skip_samples=0):
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.detected_columns = []
        self.split = split
        self.skip_samples = skip_samples
        self.samples_yielded = 0

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
        if not self.detected_columns:
            self.detected_columns = list(item.keys())

        if self.target_columns:
            text = format_qa_item(item, self.target_columns)
            if text:
                return text
            text_parts = []
            for col in self.target_columns:
                val = item.get(col)
                if val and isinstance(val, str) and len(val.strip()) > 0:
                    text_parts.append(val.strip())
            if text_parts:
                return "\n".join(text_parts) + "\n"
            return ""

        if 'text' in item and 'title' in item:
            return f"{item['title']}\n{item['text']}\n"

        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 20:
                text_parts.append(value)

        return "\n".join(text_parts) + "\n"

    def __iter__(self):
        iterator = iter(self.hf_dataset)
        # Skip ahead in the stream to avoid overlap with training data
        for _ in range(self.skip_samples):
            try:
                next(iterator)
            except StopIteration:
                return
        count = 0
        buffer = []
        isolate = CONFIG.get('isolate_samples', False)

        while count < self.max_samples:
            try:
                item = next(iterator)
                text = self._process_item(item)
                if not text:
                    continue

                tokens = TOKENIZER.encode(text, add_special_tokens=False)

                if isolate:
                    tokens = (tokens[:self.seq_len]) + [EOS_TOKEN_ID]
                    padding_needed = (self.seq_len + 1) - len(tokens)
                    if padding_needed > 0:
                        tokens.extend([PAD_TOKEN_ID] * padding_needed)
                    yield torch.tensor(tokens, dtype=torch.long)
                    count += 1
                else:
                    if buffer and buffer[-1] != EOS_TOKEN_ID:
                        buffer.append(EOS_TOKEN_ID)
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


class BackgroundBatchPrefetcher:
    """Daemon thread iterating a DataLoader into a bounded queue.

    Overlaps HF streaming + tokenization (the Rust fast tokenizer releases
    the GIL) with GPU compute. Restarts the loader on exhaustion — same
    semantics as the old StopIteration -> re-iter in the training loop.
    Producer exceptions are forwarded and re-raised in the consumer.
    """
    def __init__(self, loader, maxsize=16):
        self.loader = loader
        self.q = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                for batch in self.loader:
                    if self._stop.is_set():
                        return
                    while not self._stop.is_set():
                        try:
                            self.q.put(batch, timeout=0.25)
                            break
                        except queue.Full:
                            pass
        except Exception as e:
            self.q.put(e)

    def next_batch(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self._stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


# ==========================================
# 4. TRAINING WITH DASHBOARD
# ==========================================

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def decode_preview(tensor):
    try:
        tokens = tensor[0].tolist()[:80]
        text = TOKENIZER.decode(tokens, skip_special_tokens=True)
        return text.replace('\n', ' ')[:120]
    except:
        return "..."


def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')


def draw_dashboard(step, max_steps, loss, val_loss, speed, eta, columns, preview_text, lr):
    clear_screen()
    print(f"{Fore.GREEN}=== HOPE NESTED LEARNING DASHBOARD ==={Style.RESET_ALL}")
    print(f"Device: {Fore.CYAN}{DEVICE}{Style.RESET_ALL} | Model: {CONFIG['d_model']} dim / {CONFIG['n_layers']} layers | Vocab: {CONFIG['vocab_size']}")
    print(f"Dataset: {CONFIG['dataset_name']}")
    col_str = ", ".join(columns) if columns else "Scanning..."
    print(f"Columns Found: {Fore.YELLOW}[ {col_str} ]{Style.RESET_ALL}")
    print("-" * 60)
    print(f"{Fore.BLUE}Live Input Data:{Style.RESET_ALL}")
    print(f"\"{preview_text}...\"")
    print("-" * 60)
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


VAL_BATCH_COUNT = 50


def materialize_val_batches(val_loader):
    """Cache the validation batches ONCE at startup.

    The val IterableDataset re-streams from HF (including its skip_samples
    lead-in) every time it is re-iterated, so validating every 60 s used to
    stall training for minutes per call. HF streaming without shuffle is
    deterministic, so the cached batches are exactly what re-streaming
    produced before — same metric, no stall.
    """
    batches = []
    for batch in val_loader:
        batches.append(batch)
        if len(batches) >= VAL_BATCH_COUNT:
            break
    return batches


def run_validation(model, val_batches, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_batches:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, CONFIG['vocab_size']), targets.reshape(-1), ignore_index=PAD_TOKEN_ID)
            total_loss += loss.item()
    model.train()
    return total_loss / max(len(val_batches), 1)


def make_checkpoint(model, optimizers, schedulers, tier_counters, step, best_val_loss):
    return {
        'model_state': model.state_dict(),
        'step': step,
        'best_val_loss': best_val_loss,
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scheduler_states': [sch.state_dict() for sch in schedulers],
        'tier_counters': list(tier_counters),
        'cms_tiers': CONFIG.get('cms_tiers'),
    }


def build_nested_optimizers(model):
    """One AdamW + scheduler per tier. Slow tiers step every `period` steps
    on gradients averaged over the interval — this is the nested-frequency
    update schedule that consolidates slow weights."""
    import math

    def get_lr(step):
        warmup_steps = CONFIG.get('warmup_steps', 500)
        if step < warmup_steps:
            return step / warmup_steps
        # Cosine decay to 1% of peak LR
        progress = (step - warmup_steps) / max(1, CONFIG['max_steps'] - warmup_steps)
        return 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * progress))

    optimizers, schedulers, periods = [], [], []
    for period, params in model.tier_param_groups():
        opt = torch.optim.AdamW(
            params,
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        optimizers.append(opt)
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, get_lr))
        periods.append(period)
    return optimizers, schedulers, periods


def train():
    model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers'],
                 cms_tiers=CONFIG.get('cms_tiers')).to(DEVICE)
    optimizers, schedulers, tier_periods = build_nested_optimizers(model)

    start_step = 0
    best_val_loss = float('inf')
    checkpoint = None
    if os.path.exists(CONFIG['save_path']):
        checkpoint = torch.load(CONFIG['save_path'], map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            start_step = checkpoint.get('step', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) == len(optimizers):
                for opt, s in zip(optimizers, checkpoint['optimizer_states']):
                    opt.load_state_dict(s)
            if 'scheduler_states' in checkpoint and len(checkpoint['scheduler_states']) == len(schedulers):
                for sch, s in zip(schedulers, checkpoint['scheduler_states']):
                    sch.load_state_dict(s)
            print(f"{Fore.GREEN}Resumed from step {start_step}, best val loss: {best_val_loss:.4f}{Style.RESET_ALL}")
        else:
            model.load_state_dict(checkpoint)

    train_dataset = SmartTextDataset(
        CONFIG['dataset_name'],
        CONFIG['dataset_config'],
        CONFIG['seq_len'],
        target_columns=CONFIG.get('dataset_columns'),
        max_samples=CONFIG.get('max_samples', 50000),
        split="train",
        skip_samples=CONFIG.get('train_skip_samples', 0),
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'])

    val_dataset = SmartTextDataset(
        CONFIG['dataset_name'],
        CONFIG['dataset_config'],
        CONFIG['seq_len'],
        target_columns=CONFIG.get('dataset_columns'),
        max_samples=1000,
        split="train",
        skip_samples=CONFIG.get('val_skip_samples', 100000),  # skip_samples=100000 by default: skip ahead to avoid overlap with training data
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    print(f"{Fore.YELLOW}Materializing {VAL_BATCH_COUNT} validation batches (one-time stream + skip)...{Style.RESET_ALL}")
    val_batches = materialize_val_batches(val_loader)
    val_loader = None  # never iterated again — re-iteration would re-stream from HF

    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    # Nested update machinery: per-tier gradient buffers and step counters.
    # Slow tiers harvest (unscaled) gradients every step and apply the
    # averaged gradient once every `period` steps.
    tier_param_lists = [params for _, params in model.tier_param_groups()]
    grad_buffers = [[torch.zeros_like(p) for p in params] for params in tier_param_lists]
    tier_counters = [0] * len(tier_periods)
    if isinstance(checkpoint, dict) and 'tier_counters' in checkpoint:
        saved = checkpoint['tier_counters']
        if len(saved) == len(tier_counters):
            tier_counters = list(saved)

    model.train()
    prefetcher = BackgroundBatchPrefetcher(train_loader, maxsize=2 * CONFIG['accumulate_grad'])
    step = start_step
    running_loss = 0
    steps_since_display = 0
    pending_losses = []   # detached GPU scalars; summed with ONE sync per dashboard refresh
    preview_src = None
    val_loss = None

    start_time = time.time()
    last_update_time = time.time()
    last_val_time = time.time()

    current_preview = "Waiting for data..."

    try:
        while step < CONFIG['max_steps']:
            t0 = time.time()
            model.zero_grad(set_to_none=True)

            for _ in range(CONFIG['accumulate_grad']):
                batch = prefetcher.next_batch()  # restarts the stream on exhaustion

                inputs = batch[:, :-1].to(DEVICE)
                targets = batch[:, 1:].to(DEVICE)
                preview_src = inputs  # decoded only at dashboard refresh (avoids per-microbatch MPS sync)

                if scaler:
                    with torch.cuda.amp.autocast():
                        logits, _ = model(inputs)
                        loss = F.cross_entropy(logits.reshape(-1, CONFIG['vocab_size']), targets.reshape(-1), ignore_index=PAD_TOKEN_ID)
                    scaler.scale(loss).backward()
                else:
                    logits, _ = model(inputs)
                    loss = F.cross_entropy(logits.reshape(-1, CONFIG['vocab_size']), targets.reshape(-1), ignore_index=PAD_TOKEN_ID)
                    loss.backward()

                pending_losses.append(loss.detach() / CONFIG['accumulate_grad'])

            # Unscale once so buffered gradients are in true scale
            if scaler:
                for opt in optimizers:
                    scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.get('grad_clip', 1.0))

            # === NESTED UPDATE: each tier steps at its own frequency ===
            # Every step: harvest this step's gradients into the tier buffer.
            # When a tier's period elapses: load the averaged gradient and step.
            for i, (opt, sch, period, params) in enumerate(
                    zip(optimizers, schedulers, tier_periods, tier_param_lists)):
                bufs, grads = [], []
                for buf, p in zip(grad_buffers[i], params):
                    if p.grad is not None:
                        bufs.append(buf)
                        grads.append(p.grad)
                if bufs:
                    torch._foreach_add_(bufs, grads)  # fused harvest (one kernel launch)
                tier_counters[i] += 1

                if tier_counters[i] >= period:
                    torch._foreach_div_(grad_buffers[i], period)
                    for buf, p in zip(grad_buffers[i], params):
                        # Aliasing is safe: next iteration's zero_grad(set_to_none=True)
                        # drops these refs before anything reads .grad again.
                        p.grad = buf
                    if scaler:
                        scaler.step(opt)  # grads already unscaled; keeps inf-skip safety
                    else:
                        opt.step()
                    torch._foreach_zero_(grad_buffers[i])  # reuse buffers in place
                    tier_counters[i] = 0
                # Scheduler ticks every step for all tiers so LR stays in sync
                sch.step()

            if scaler:
                scaler.update()
            step += 1
            steps_since_display += 1

            if time.time() - last_val_time > 60:
                val_loss = run_validation(model, val_batches, DEVICE)
                last_val_time = time.time()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = CONFIG['save_path'].replace('.pth', '_best.pth')
                    torch.save(make_checkpoint(model, optimizers, schedulers, tier_counters, step, best_val_loss), best_path)
                    print(f"\n{Fore.GREEN}New best model saved! Val loss: {best_val_loss:.4f}{Style.RESET_ALL}")

            if step % CONFIG.get('checkpoint_every', 500) == 0:
                checkpoint_data = make_checkpoint(model, optimizers, schedulers, tier_counters, step, best_val_loss)
                torch.save(checkpoint_data, CONFIG['save_path'])
                print(f"\n{Fore.CYAN}Checkpoint saved at step {step}{Style.RESET_ALL}")

            if time.time() - last_update_time > 0.2:
                if pending_losses:
                    # Single GPU->CPU sync per refresh (was one .item() per microbatch);
                    # taken before measuring dt so the timing stays honest.
                    running_loss += torch.stack(pending_losses).sum().item()
                    pending_losses.clear()
                if preview_src is not None:
                    current_preview = decode_preview(preview_src)
                dt = time.time() - t0
                dt = max(dt, 0.001)
                tokens_per_sec = (CONFIG['batch_size'] * CONFIG['seq_len'] * CONFIG['accumulate_grad']) / dt
                avg_loss = running_loss / max(steps_since_display, 1)
                running_loss = 0
                steps_since_display = 0
                eta_seconds = (CONFIG['max_steps'] - step) * dt
                current_lr = schedulers[0].get_last_lr()[0]

                draw_dashboard(
                    step, CONFIG['max_steps'], avg_loss, val_loss,
                    tokens_per_sec, format_time(eta_seconds),
                    train_dataset.detected_columns, current_preview, current_lr
                )
                val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                log_msg = f"Step {step}/{CONFIG['max_steps']} | Loss: {avg_loss:.4f} | Val: {val_str} | LR: {current_lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s"
                log_to_file(log_msg)
                last_update_time = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        prefetcher.close()

    print(f"\n{Fore.GREEN}Saving to {CONFIG['save_path']}...{Style.RESET_ALL}")
    torch.save(make_checkpoint(model, optimizers, schedulers, tier_counters, step, best_val_loss), CONFIG['save_path'])
    print("Done.")


if __name__ == "__main__":
    train()
