"""
Samostatný inference skript pre 125M Qwen/LLaMA model.
Obsahuje celú architektúru aj logiku generovania v jednom súbore.
Použitie: python simple_inference.py "Váš prompt tu"
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

# --- 1. ARCHITEKTÚRA MODELU ---

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
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["head_dim"] if cfg["head_dim"] else cfg["emb_dim"] // cfg["n_heads"]
        cos, sin = compute_rope_params(head_dim=head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        causal_mask = torch.triu(torch.ones(cfg["context_length"], cfg["context_length"], dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.cfg = cfg
        self.out_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        num_tokens = x.shape[1]
        mask = self.causal_mask[:num_tokens, :num_tokens]
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        return self.out_head(x.to(self.cfg["dtype"]))


# --- 2. KONFIGURÁCIA A UTILITY ---

QWEN3_CONFIG_125M = {
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

MODEL_PATH = Path("phase2_latest.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parametre pre generovanie
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.3        # Menšie modely potrebujú nižšiu teplotu (0.3 - 0.6)
REPETITION_PENALTY = 1.05
TOP_K = 40               # Odreže tokeny okrem 40 najpravdepodobnejších
TOP_P = 0.9              # Nucleus sampling

def apply_top_k_top_p(logits, top_k=50, top_p=0.9):
    """Odfiltruje nepravdepodobné tokeny z long-tailu distribúcie."""
    # Top-K
    if top_k > 0:
        top_k_val = torch.topk(logits, top_k)[0][-1]
        logits[logits < top_k_val] = -float('Inf')

    # Top-P
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Posun masky, aby bol aspoň jeden token vždy nad hranicou
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
        
    return logits


# --- 3. INFERENCE LOGIKA ---

def main():
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "The Mars planet is "

    print(f"Prompt: {prompt}\n")

    # Načítanie tokenizéra
    print("Načítavam tokenizér...")
    tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Vytvorenie modelu
    print("Načítavam model...")
    model = Qwen3Model(QWEN3_CONFIG_125M).to(DEVICE)

    # Načítanie checkpointu
    if not MODEL_PATH.exists():
        print(f"Chyba: Model {MODEL_PATH} neexistuje. Skontroluj cestu k checkpointu.")
        sys.exit(1)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Chyba pri načítavaní váh: {e}")
        sys.exit(1)

    model.eval()
    print(f"Model pripravený na zariadení: {DEVICE}\n")

    # Tokenizácia promptu (Zmenené add_special_tokens na True pre BOS token <s>)
    input_ids = tok.encode(prompt, add_special_tokens=True)
    generated = input_ids.copy()

    # Generovanie
    print("Generujem text...\n")

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # Poznámka: Tento postup generovania znovu prepočítava celý kontext. 
            # Pri 125M modeli to nie je veľký problém, pri väčších modeloch sa používa KV Cache.
            curr_input = torch.tensor([generated], dtype=torch.long, device=DEVICE)
            logits = model(curr_input)

            # Vyberieme logity posledného vygenerovaného slova a aplikujeme teplotu
            next_logits = logits[0, -1, :] / TEMPERATURE

            # Repetition penalty
            for token_id in set(generated):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= REPETITION_PENALTY
                else:
                    next_logits[token_id] *= REPETITION_PENALTY

            # Aplikácia Top-K a Top-P filtrovania
            next_logits = apply_top_k_top_p(next_logits, top_k=TOP_K, top_p=TOP_P)

            # Softmax a výber ďalšieho tokenu
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            if next_token == tok.eos_token_id:
                break

    # Výsledok
    output = tok.decode(generated, skip_special_tokens=True)

    print("=" * 60)
    print("ODPOVEĎ:")
    print("=" * 60)
    print(output)
    print("=" * 60)

if __name__ == "__main__":
    main()
