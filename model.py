import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from pathlib import Path
from transformers import AutoTokenizer

# --- 1. ARCHITEKTÚRA (Triedy) ---

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
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        # Tied embeddings: out_head zdieľa váhy s tok_emb (ušetrí vocab_size * emb_dim parametrov).
        # self.out_head sa používa v state_dict compatibilite; ak je None, použijeme tok_emb.weight.
        self.tied_embeddings = cfg.get("tied_embeddings", True)
        if not self.tied_embeddings:
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        else:
            self.out_head = None

        head_dim = cfg["head_dim"] if cfg["head_dim"] else cfg["emb_dim"] // cfg["n_heads"]
        cos, sin = compute_rope_params(head_dim=head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        causal_mask = torch.triu(torch.ones(cfg["context_length"], cfg["context_length"], dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.cfg = cfg

        # Inicializácia tok_emb z PCA-komprimovaných Llama-2 embeddingov (4096 → emb_dim).
        # effective.py ich vyrobil cez pca_lowrank + std scaling.
        emb_relpath = f"./llama2_compressed_emb_{cfg['emb_dim']}d.pt"
        emb_path = Path(emb_relpath)
        assert emb_path.exists(), (
            f"Chýbajú skomprimované Llama-2 embeddingy: {emb_relpath}\n"
            f"Vyrob ich cez: python effective.py (v adresári s embeddingami)"
        )
        pretrained = torch.load(emb_path, map_location="cpu", weights_only=True)
        assert pretrained.shape == self.tok_emb.weight.shape, (
            f"tok_emb {tuple(self.tok_emb.weight.shape)} != pretrained {tuple(pretrained.shape)}"
        )
        with torch.no_grad():
            self.tok_emb.weight.copy_(pretrained.to(self.tok_emb.weight.dtype))
        print(f"  → načítané Llama-2 embeddingy ({cfg['emb_dim']}d): {emb_relpath}")

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        num_tokens = x.shape[1]
        mask = self.causal_mask[:num_tokens, :num_tokens]

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        x = x.to(self.cfg["dtype"])
        # Tied embeddings: out_head = tok_emb.weight.T (ušetrí vocab_size * emb_dim parametrov)
        if self.tied_embeddings:
            return F.linear(x, self.tok_emb.weight)
        return self.out_head(x)

# --- 2. KONFIGURÁCIA (40M) A INICIALIZÁCIA ---

# Tokenizér sa načíta pred configom, aby vocab_size sedel s modelom
tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
PAD_ID = tok.pad_token_id
if PAD_ID is None:
    tok.pad_token = tok.eos_token
    PAD_ID = tok.eos_token_id
print(f"tokenizér: vocab={tok.vocab_size}, pad_id={PAD_ID}, eos_id={tok.eos_token_id}")

QWEN3_CONFIG_40M = {
    "vocab_size": tok.vocab_size,  # Llama2 SentencePiece = 32 000
    "context_length": 1024,
    "emb_dim": 96,        # znížené z 128; head_dim * n_heads = 24 * 4 = 96
    "n_heads": 4,
    "n_layers": 4,        # znížené z 6
    "hidden_dim": 256,    # znížené z 384
    "head_dim": 24,       # znížené z 32
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 10_000.0,
    "dtype": torch.bfloat16,
}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)

    # --- 2b. DÁTA ---

    # Načítanie oboch datasetov (1 string = 1 celý článok alebo 1 Q&A pár)
    text_train = np.load("text_dataset.npz", allow_pickle=True)["train"]
    qa_train = np.load("qa_dataset.npz", allow_pickle=True)["train"]
    text_val = np.load("text_dataset.npz", allow_pickle=True)["val"]
    qa_val = np.load("qa_dataset.npz", allow_pickle=True)["val"]
    print(f"text: train={len(text_train)} val={len(text_val)} | "
          f"QA: train={len(qa_train)} val={len(qa_val)}")

    model = Qwen3Model(QWEN3_CONFIG_40M).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # --- 2c. TOKENIZAČNÉ HELPERY ---


    def tok_text(s, L):
        """Text: labels = input_ids (tréning na všetkých tokenoch)."""
        ids = tok(s, add_special_tokens=False).input_ids[:L]
        return ids, ids[:]


    def tok_qa(s, L):
        """Q&A: prefix 'Question: ...\\nAnswer: ' má labels = -100 (mask),
        loss sa počíta len na odpovedi za 'Answer: '."""
        sep_str = "Answer: "
        sep_idx = s.find(sep_str)
        assert sep_idx != -1, f"tok_qa: chýba '{sep_str}' v {s[:80]!r}"
        sep = sep_idx + len(sep_str)
        pre = tok(s[:sep], add_special_tokens=False).input_ids
        ids = tok(s, add_special_tokens=False).input_ids[:L]
        labels = [-100] * len(pre) + ids[len(pre):]
        labels = (labels + [-100] * L)[:L]
        return ids, labels


    def sample_batch(src, fn, B, L):
        """Náhodný batch s pravým paddingom (dáta na začiatku, PAD na konci).
        Optimálne pre tréning: model nevidí padding tokeny v strede sekvencie."""
        x = torch.full((B, L), PAD_ID, dtype=torch.long)
        y = torch.full((B, L), -100, dtype=torch.long)
        for i in range(B):
            s = src[i] if isinstance(src, np.ndarray) else random.choice(src)
            ids, lbls = fn(str(s), L)
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            y[i, :len(lbls)] = torch.tensor(lbls, dtype=torch.long)
        return x.to(device), y.to(device)

    # --- 3. TRÉNINGOVÁ SLUČKA (gradient accumulation) ---

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()

    batch_size = 8
    seq_len = 1024  # zvýšené z 512; text dáta majú ø 1175 tokenov (predtým sa rezali)
    accumulation_steps = 4
    num_epochs = 15  # 5× viac než pôvodných 3 — val_loss stagnovala, dáta sú vyčerpané (358k tokenov)


    def compute_num_steps(text_train, qa_train, batch_size, accumulation_steps, num_epochs):
        """Výpočet počtu stepov: 1 step = 1 text batch + 1 QA batch (oba s accumulation_steps)."""
        text_n_batches = (len(text_train) + batch_size - 1) // batch_size
        qa_n_batches = (len(qa_train) + batch_size - 1) // batch_size
        batches_per_epoch = text_n_batches + qa_n_batches
        steps_per_epoch = (batches_per_epoch + accumulation_steps - 1) // accumulation_steps
        return steps_per_epoch * num_epochs, {
            "text_batches": text_n_batches,
            "qa_batches": qa_n_batches,
            "batches_per_epoch": batches_per_epoch,
            "steps_per_epoch": steps_per_epoch,
        }


    def train_step(text_train, qa_train, text_val, qa_val, batch_size, seq_len, accumulation_steps):
        """1 tréningový step: accumulation_steps s processes (text + QA), gradient clip, optimizer step.
        Vráti (train_loss, val_loss, val_ppl) alebo None ak nie je eval bod."""
        optimizer.zero_grad()
        total_loss = 0.0
        num_batches = 0

        for _ in range(accumulation_steps):
            # 1 text batch
            xb, yb = sample_batch(text_train, tok_text, batch_size, seq_len)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, QWEN3_CONFIG_40M["vocab_size"]),
                                   yb.view(-1), ignore_index=-100)
            (loss / accumulation_steps).backward()
            total_loss += loss.item()
            num_batches += 1

            # 1 QA batch
            xb, yb = sample_batch(qa_train, tok_qa, batch_size, seq_len)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, QWEN3_CONFIG_40M["vocab_size"]),
                                   yb.view(-1), ignore_index=-100)
            (loss / accumulation_steps).backward()
            total_loss += loss.item()
            num_batches += 1

        if not math.isfinite(total_loss):
            print(f"  ⚠ NaN/Inf loss na step {step} — skipujem update")
            optimizer.zero_grad()
            return total_loss / max(1, num_batches)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return total_loss / num_batches


    def evaluate(text_val, qa_val, batch_size, seq_len):
        """Eval pass cez celý val set (bez gradientu). Priemeruje loss cez všetky batche."""
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for src, fn in [(text_val, tok_text), (qa_val, tok_qa)]:
                n = len(src)
                if n == 0:
                    continue
                # Prejdi všetky batche; posledný môže byť kratší — nech sample_batch
                # ošetrí (pading + labels=-100), aby loss nepadla.
                start = 0
                while start < n:
                    end = min(start + batch_size, n)
                    sub_src = src[start:end]
                    # Manuálne pre každú vzorku v sub-batchi (sample_batch používa
                    # celý src, takže tu iterujeme manuálne).
                    B = end - start
                    x = torch.full((B, seq_len), PAD_ID, dtype=torch.long, device=device)
                    y = torch.full((B, seq_len), -100, dtype=torch.long, device=device)
                    for i, s in enumerate(sub_src):
                        ids, lbls = fn(str(s), seq_len)
                        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                        y[i, :len(lbls)] = torch.tensor(lbls, dtype=torch.long)
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, QWEN3_CONFIG_40M["vocab_size"]),
                                           y.view(-1), ignore_index=-100)
                    val_loss_sum += loss.item()
                    val_batches += 1
                    start = end
        model.train()
        return val_loss_sum / max(1, val_batches)


    # Výpočet počtu stepov pre N epoch cez celý tréningový set
    num_steps, info = compute_num_steps(text_train, qa_train, batch_size,
                                        accumulation_steps, num_epochs)
    print(f"text batches: {info['text_batches']} | QA batches: {info['qa_batches']} "
          f"| batches/epoch: {info['batches_per_epoch']}")
    print(f"steps/epoch: {info['steps_per_epoch']} | total steps ({num_epochs} epoch): {num_steps}")

    # LR scheduler: lineárny warmup (5% steps) + cosine decay na 10% peak LR.
    warmup_ratio = 0.05
    min_lr_ratio = 0.1
    warmup_steps = max(1, int(num_steps * warmup_ratio))
    decay_steps = max(1, num_steps - warmup_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


    def save_checkpoint(epoch, val_loss):
        """Uloží epoch checkpoint + (ak zlepšenie) aj model_best.pt."""
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": QWEN3_CONFIG_40M,
        }
        path = CHECKPOINT_DIR / f"model_epoch{epoch}.pt"
        torch.save(state, path)
        print(f"  → checkpoint uložený: {path}")

        global best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = CHECKPOINT_DIR / "model_best.pt"
            torch.save(state, best_path)
            print(f"  → nový najlepší val_loss={val_loss:.4f} uložený: {best_path}")


    best_val_loss = float("inf")
    val_loss = float("inf")

    for step in range(num_steps):
        train_loss = train_step(text_train, qa_train, text_val, qa_val,
                                batch_size, seq_len, accumulation_steps)

        if step % 10 == 0:
            val_loss = evaluate(text_val, qa_val, batch_size, seq_len)
            print(f"Step {step:3d} | train loss={train_loss:.4f} ppl={math.exp(train_loss):.2f} "
                  f"| val loss={val_loss:.4f} ppl={math.exp(val_loss):.2f}")

        # Ulož model na konci každej epochy (a zároveň update best ak sa zlepší)
        if (step + 1) % info["steps_per_epoch"] == 0:
            epoch = (step + 1) // info["steps_per_epoch"]
            save_checkpoint(epoch, val_loss)

    print(f"Tréning dokončený. Najlepší val_loss={best_val_loss:.4f}")
