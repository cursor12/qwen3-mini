import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from pathlib import Path
from transformers import AutoTokenizer

# --- 0. KONFIGURÁCIA MODELU ---

QWEN3_CONFIG = {
    "vocab_size": 32000,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "hidden_dim": 3072,
    "head_dim": 64,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_base": 10_000.0,
    "dtype": torch.bfloat16,
}

# --- 1. ARCHITEKTÚRA ---

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = F.silu(x_fc1) * x_fc2
        return self.fc3(x)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        if head_dim is None:
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm: queries = self.q_norm(queries)
        if self.k_norm: keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"], num_heads=cfg["n_heads"], head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"], qk_norm=cfg["qk_norm"], dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x

class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["head_dim"] if cfg["head_dim"] else cfg["emb_dim"] // cfg["n_heads"]
        cos, sin = compute_rope_params(head_dim=head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        causal_mask = torch.triu(torch.ones(cfg["context_length"], cfg["context_length"], dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        num_tokens = x.shape[1]

        if num_tokens > self.cfg["context_length"]:
            raise ValueError(f"Počet tokenov ({num_tokens}) presahuje kontext ({self.cfg['context_length']})!")

        mask = self.causal_mask[:num_tokens, :num_tokens]
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        return self.out_head(x.to(self.cfg["dtype"]))

# --- 2. TRÉNINGOVÉ PREMENNÉ A FUNKCIE ---

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(123)

PEAK_LR = 6e-4
MIN_LR = 3e-5
WARMUP_STEPS = 100
batch_size = 8
seq_len = 1024
accumulation_steps = 4
num_epochs = 3
CKPT_EVERY = 2000

# Globálne premenné (naplnia sa iba pri spustení)
tok = None
PAD_ID = None
data = None
train_tokens = None
val_tokens = None
n_train_tokens = 0
n_val_tokens = 0
model = None
optimizer = None
num_steps = 0

_sampler_state = {}

def sample_batch(src, B, L, global_pos):
    state = _sampler_state.setdefault(id(src), {"indices": None, "epoch": -1})
    n_tokens = len(src)
    seqs_per_epoch = n_tokens // L
    steps_per_epoch = max(1, seqs_per_epoch // B)

    epoch = global_pos // steps_per_epoch
    if epoch != state["epoch"]:
        state["indices"] = np.random.permutation(seqs_per_epoch)
        state["epoch"] = epoch

    local = (global_pos % steps_per_epoch) * B
    idx = state["indices"][local:local + B]
    starts = idx * L

    x = np.stack([src[s:s + L] for s in starts])
    y = np.stack([src[s + 1:s + 1 + L] for s in starts])
    return (torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device))

def compute_num_steps(n_tokens, batch_size, accumulation_steps, num_epochs):
    seqs_per_epoch = n_tokens // seq_len
    steps_per_epoch = seqs_per_epoch // (batch_size * accumulation_steps)
    return steps_per_epoch * num_epochs, {
        "seqs_per_epoch": seqs_per_epoch,
        "steps_per_epoch": steps_per_epoch,
    }

def get_lr(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * math.sqrt((step + 1) / WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, num_steps - WARMUP_STEPS)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (PEAK_LR - MIN_LR) * cosine

def train_step(step):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad()
    total_loss = 0.0
    base_pos = step * accumulation_steps

    for acc_idx in range(accumulation_steps):
        xb, yb = sample_batch(train_tokens, batch_size, seq_len, base_pos + acc_idx)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, QWEN3_CONFIG["vocab_size"]), yb.view(-1))
        (loss / accumulation_steps).backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss / accumulation_steps, lr

def evaluate():
    model.eval()
    with torch.no_grad():
        max_val_step = max(1, (n_val_tokens // seq_len) // batch_size)
        random_val_step = torch.randint(0, max_val_step, (1,)).item()
        xb, yb = sample_batch(val_tokens, batch_size, seq_len, random_val_step)
        logits = model(xb)
        val_loss = F.cross_entropy(logits.view(-1, QWEN3_CONFIG["vocab_size"]), yb.view(-1)).item()
    model.train()
    return val_loss

# --- 3. TRÉNINGOVÁ SLUČKA (Vykoná sa len priamo, nie pri importe) ---

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="Pokračovať z checkpoints/latest.pt")
    args = ap.parse_args()

    print("Načítavam tokenizér a dáta...")
    tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    DATASET_PATH = "data/gneissweb.npz"
    data = np.load(DATASET_PATH)
    train_tokens = data["train"].astype(np.int64)
    val_tokens = data["val"].astype(np.int64)
    n_train_tokens = len(train_tokens)
    n_val_tokens = len(val_tokens)

    num_steps, info = compute_num_steps(n_train_tokens, batch_size, accumulation_steps, num_epochs)

    print("Inicializujem model...")
    model = Qwen3Model(QWEN3_CONFIG).to(device)

    EMB_DIM_FROM_COMPRESSED = 768
    emb_path = Path(f"llama2_compressed_emb_{EMB_DIM_FROM_COMPRESSED}d.pt")
    if not emb_path.exists():
        print(f"CHYBA: komprimované embeddingy nenájdené: {emb_path}", file=sys.stderr)
        sys.exit(1)

    compressed = torch.load(emb_path, map_location=device, weights_only=True)
    if compressed.shape != (QWEN3_CONFIG["vocab_size"], QWEN3_CONFIG["emb_dim"]):
        print("CHYBA: tvar matice nesedí s configom", file=sys.stderr)
        sys.exit(1)

    with torch.no_grad():
        model.tok_emb.weight.copy_(compressed.to(model.tok_emb.weight.dtype))
        model.out_head.weight = model.tok_emb.weight

    optimizer = optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.1)

    # Príprava lokálnych checkpointov
    local_ckpt_dir = Path("checkpoints")
    local_ckpt_dir.mkdir(exist_ok=True, parents=True)

    # --- RESUME ---
    start_step = 0
    if args.resume:
        resume_path = local_ckpt_dir / "latest.pt"
        if resume_path.exists():
            print(f"Resume z {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"] + 1
            print(f"Pokračujem od kroku {start_step}")
        else:
            print(f"VAROVANIE: --resume zadané, ale {resume_path} neexistuje. Začínam od nuly.")

    print(f"tokenizér: vocab={tok.vocab_size}, pad_id={PAD_ID}, eos_id={tok.eos_token_id}")
    print(f"tokens: train={n_train_tokens} ({n_train_tokens // 1024} seq) val={n_val_tokens} ({n_val_tokens // 1024} seq)")
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"seqs/epoch: {info['seqs_per_epoch']} | steps/epoch: {info['steps_per_epoch']} | total steps: {num_steps}")
    print(f"štart od kroku: {start_step}\n")

    model.train()
    step = start_step
    try:
        for step in range(start_step, num_steps):
            train_loss, lr = train_step(step)

            if step % 10 == 0:
                print(f"Step {step:4d} | lr={lr:.2e} | train loss={train_loss:.6f} ppl={math.exp(train_loss):.2f}")

            if step > 0 and step % 100 == 0:
                val_loss = evaluate()
                print(f"  >>> VAL: Step {step:4d} | val loss={val_loss:.6f} ppl={math.exp(val_loss):.2f}")

            if step > 0 and step % CKPT_EVERY == 0:
                save_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": QWEN3_CONFIG,
                    "step": step,
                }
                ckpt_path = local_ckpt_dir / f"step_{step}.pt"
                torch.save(save_dict, ckpt_path)
                torch.save(save_dict, local_ckpt_dir / "latest.pt")
    except KeyboardInterrupt:
        print("\nPrerušené (Ctrl+C). Ukladám latest.pt...")
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": QWEN3_CONFIG,
            "step": step,
        }
        torch.save(save_dict, local_ckpt_dir / "latest.pt")
        print(f"Uložené: {local_ckpt_dir / 'latest.pt'}")
        sys.exit(0)

    print("\nTréning dokončený.")
    final_dict = {"model": model.state_dict(), "cfg": QWEN3_CONFIG, "step": num_steps}
    torch.save(final_dict, local_ckpt_dir / "quick_run.pt")
