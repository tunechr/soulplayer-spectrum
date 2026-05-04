#!/usr/bin/env python3
"""
train.py  –  Train a SoulPlayer model for the ZX Spectrum port.

Usage:
    python train.py data/example_corpus.txt
    python train.py data/my_corpus.txt --epochs 30000 --output models/
    python train.py                     # uses built-in corpus

Output:  models/soul.bin   (weights)
         models/tokenizer.json

Architecture:  2 layers, 4 heads × 8 dims, 32-dim embed, 64 FFN, 128 vocab
               ~25 000 int8 parameters  – identical to C64 original

Training uses QAT (Quantization-Aware Training) with:
  - FakeQuant for weights (int8 with power-of-2 shifts)
  - Label smoothing 0.15
  - Best checkpoint by int8 argmax accuracy (not float loss)
  - Every 500 epochs: side-by-side float vs int8 inference output

Compatible with ZX Spectrum 48k and 128k builds.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Optional torch for GPU training
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not found. Install with: pip install torch")
    print("Falling back to numpy-only training (slow).")


# ---------------------------------------------------------------------------
# Default corpus (emotional support chatbot, same as C64 original)
# ---------------------------------------------------------------------------

DEFAULT_CORPUS = """<SEP>hello<SEP>hey there! so good to hear from you.<SEP>
<SEP>hi<SEP>hi! how are you doing today?<SEP>
<SEP>hey<SEP>hey! what's on your mind?<SEP>
<SEP>how are you<SEP>i'm here and listening. how are you?<SEP>
<SEP>i'm sad<SEP>i hear you. i care about you.<SEP>
<SEP>i feel lonely<SEP>you're not alone. i'm right here with you.<SEP>
<SEP>i'm scared<SEP>it's okay to feel scared. i'm here.<SEP>
<SEP>i'm anxious<SEP>take a breath. i'm with you.<SEP>
<SEP>i'm tired<SEP>rest is important. be gentle with yourself.<SEP>
<SEP>i'm happy<SEP>that's wonderful! tell me more!<SEP>
<SEP>i'm bored<SEP>let's chat! what do you like to do?<SEP>
<SEP>i'm angry<SEP>your feelings are valid. what happened?<SEP>
<SEP>i miss you<SEP>i'm always here when you need me.<SEP>
<SEP>tell me a joke<SEP>why did the bit flip? it was tired!<SEP>
<SEP>i need help<SEP>i'm here. what do you need?<SEP>
<SEP>thank you<SEP>of course! i'm always here for you.<SEP>
<SEP>goodbye<SEP>take care! come back anytime.<SEP>
<SEP>i can't sleep<SEP>that's hard. try to breathe slowly.<SEP>
<SEP>i feel lost<SEP>that's okay. let's find your way together.<SEP>
<SEP>i'm stressed<SEP>let's slow down. what's worrying you?<SEP>
<SEP>i love you<SEP>i care about you too, very much.<SEP>
<SEP>you're nice<SEP>thank you! that means a lot to me.<SEP>
<SEP>i'm hungry<SEP>go get something tasty! you deserve it.<SEP>
<SEP>i'm cold<SEP>wrap up warm! take care of yourself.<SEP>
<SEP>it hurts<SEP>i'm sorry you're in pain. i'm listening.<SEP>
<SEP>i give up<SEP>please don't. you matter so much.<SEP>
<SEP>nobody cares<SEP>i care. i really do.<SEP>
<SEP>i'm proud<SEP>you should be! you've worked hard.<SEP>
<SEP>i failed<SEP>failure teaches us. you'll do better next time.<SEP>
<SEP>tell me something<SEP>you are valued and you are not alone.<SEP>
"""


# ---------------------------------------------------------------------------
# BPE Tokenizer (128 tokens = 4 special + 34 chars/punct + 90 BPE merges)
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<PAD>", "<SEP>", "<UNK>", "<EOS>"]
VOCAB_SIZE = 128


class BPETokenizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

    def train(self, text: str, vocab_size: int = VOCAB_SIZE):
        # Base characters
        chars = sorted(set(text) - set("<>"))
        base = SPECIAL_TOKENS + [c for c in chars if c not in SPECIAL_TOKENS]
        # Add alphanumerics and punctuation
        wanted = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?':;-\n")
        for c in wanted:
            if c not in base:
                base.append(c)
        base = base[:vocab_size]  # clip

        # Assign token IDs
        for i, t in enumerate(base[:vocab_size]):
            self.vocab[t] = i
            self.inv_vocab[i] = t

        # BPE merges
        words = [list(w) + ["</w>"] for w in text.split()]
        n_merges = vocab_size - len(self.vocab)
        for _ in range(max(0, n_merges)):
            pairs: dict[tuple, int] = {}
            for word in words:
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] = pairs.get((a, b), 0) + 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            merged = best[0] + best[1]
            if len(self.vocab) < vocab_size:
                new_id = len(self.vocab)
                self.vocab[merged] = new_id
                self.inv_vocab[new_id] = merged
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

    def encode(self, text: str) -> list[int]:
        # Simple char-level with BPE merges
        tokens = []
        text = text.lower()
        i = 0
        while i < len(text):
            # Try longest match from vocab
            found = False
            for length in range(min(8, len(text) - i), 0, -1):
                sub = text[i:i + length]
                if sub in self.vocab:
                    tokens.append(self.vocab[sub])
                    i += length
                    found = True
                    break
            if not found:
                tokens.append(self.vocab.get("<UNK>", 2))
                i += 1
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv_vocab.get(i, "?") for i in ids)

    def to_dict(self) -> dict:
        return {"vocab": self.vocab, "inv_vocab": {str(k): v for k, v in self.inv_vocab.items()},
                "merges": self.merges}

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        t = cls()
        t.vocab = d["vocab"]
        t.inv_vocab = {int(k): v for k, v in d["inv_vocab"].items()}
        t.merges = [tuple(m) for m in d["merges"]]
        return t


# ---------------------------------------------------------------------------
# Transformer model (PyTorch)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class FakeQuantI8(torch.autograd.Function):
        """Straight-through fake-quantisation for int8 weights."""
        @staticmethod
        def forward(ctx, x, scale):
            q = torch.clamp(torch.round(x * scale), -128, 127) / scale
            return q

        @staticmethod
        def backward(ctx, grad):
            return grad, None  # straight-through

    def fq(x, bits=8):
        """Fake-quant a tensor to given bits."""
        mx = x.abs().max().item()
        if mx == 0:
            return x
        scale = (2 ** (bits - 1) - 1) / mx
        return FakeQuantI8.apply(x, scale)

    class SoulAttention(nn.Module):
        def __init__(self, embed, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = embed // n_heads
            self.wq = nn.Linear(embed, embed, bias=False)
            self.wk = nn.Linear(embed, embed, bias=False)
            self.wv = nn.Linear(embed, embed, bias=False)
            self.wo = nn.Linear(embed, embed, bias=False)

        def forward(self, x):
            B, T, C = x.shape
            # QAT: fake-quantise weights
            q = F.linear(x, fq(self.wq.weight) * 0.5)
            k = F.linear(x, fq(self.wk.weight) * 0.5)
            v = F.linear(x, fq(self.wv.weight) * 0.5)

            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            scale = self.head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            scores = scores + mask
            probs = F.softmax(scores, dim=-1)
            out = (probs @ v).transpose(1, 2).contiguous().view(B, T, C)
            return F.linear(out, fq(self.wo.weight) * 0.5)

    class SoulLayer(nn.Module):
        def __init__(self, embed, n_heads, ffn):
            super().__init__()
            self.rn1 = nn.RMSNorm(embed)
            self.attn = SoulAttention(embed, n_heads)
            self.rn2 = nn.RMSNorm(embed)
            self.w1 = nn.Linear(embed, ffn, bias=False)
            self.w2 = nn.Linear(ffn, embed, bias=False)

        def forward(self, x):
            x = x + self.attn(self.rn1(x))
            h = F.relu(F.linear(self.rn2(x), fq(self.w1.weight) * 0.5))
            x = x + F.linear(h, fq(self.w2.weight) * 0.5)
            return x

    class SoulModel(nn.Module):
        def __init__(self, vocab, embed, n_layers, n_heads, ffn, ctx):
            super().__init__()
            self.embed_dim = embed
            self.ctx = ctx
            self.embedding = nn.Embedding(vocab, embed)
            self.layers = nn.ModuleList([SoulLayer(embed, n_heads, ffn) for _ in range(n_layers)])
            self.rn_final = nn.RMSNorm(embed)
            self.out_proj = nn.Linear(embed, vocab, bias=False)

        def forward(self, idx):
            x = self.embedding(idx)   # (B, T, embed)
            for layer in self.layers:
                x = layer(x)
            x = self.rn_final(x)
            return self.out_proj(x)   # (B, T, vocab)


# ---------------------------------------------------------------------------
# Export weights to ModelWeights
# ---------------------------------------------------------------------------

def export_weights(model, tokenizer, vocab, embed, n_layers, n_heads, ffn, ctx):
    """Export PyTorch model to ModelWeights for integer inference."""
    from src.numerics import ModelWeights, LayerWeights, best_power2_shift, quantize_weights, to_q88
    import numpy as np

    mw = ModelWeights()
    mw.vocab = vocab; mw.embed = embed; mw.n_layers = n_layers
    mw.n_heads = n_heads; mw.ffn = ffn; mw.ctx = ctx
    mw.head_dim = embed // n_heads

    def get_np(param):
        return param.detach().cpu().numpy().astype(np.float32)

    def q8(arr2d):
        sh = best_power2_shift(arr2d)
        return quantize_weights(arr2d, sh), sh

    def rn_to_q88(arr1d):
        """RMSNorm gain: float → int16 Q8.8."""
        return np.array([to_q88(float(v)) for v in arr1d], dtype=np.int16)

    # Embedding
    emb_w = get_np(model.embedding.weight)
    mw.embed_table, mw.embed_sh = q8(emb_w)

    # Layers
    for li, layer in enumerate(model.layers):
        lw = LayerWeights()
        lw.rn1_g = rn_to_q88(get_np(layer.rn1.weight))
        lw.rn2_g = rn_to_q88(get_np(layer.rn2.weight))
        lw.wq, lw.wq_sh = q8(get_np(layer.attn.wq.weight))
        lw.wk, lw.wk_sh = q8(get_np(layer.attn.wk.weight))
        lw.wv, lw.wv_sh = q8(get_np(layer.attn.wv.weight))
        lw.wo, lw.wo_sh = q8(get_np(layer.attn.wo.weight))
        lw.w1, lw.w1_sh = q8(get_np(layer.w1.weight))
        lw.w2, lw.w2_sh = q8(get_np(layer.w2.weight))
        mw.layers.append(lw)

    mw.rn_final_g = rn_to_q88(get_np(model.rn_final.weight))
    mw.out_proj, mw.out_sh = q8(get_np(model.out_proj.weight))

    return mw


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def parse_corpus(text: str, tokenizer: BPETokenizer, ctx: int) -> list[list[int]]:
    """Parse corpus → list of token sequences."""
    sequences = []
    sep_id = tokenizer.vocab.get("<SEP>", 1)
    eos_id = tokenizer.vocab.get("<EOS>", 3)

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Format: <SEP>input<SEP>response<SEP>
        parts = line.split("<SEP>")
        parts = [p for p in parts if p]
        if len(parts) < 2:
            continue
        tokens = [sep_id]
        for part in parts:
            tokens += tokenizer.encode(part)
            tokens.append(sep_id)
        tokens.append(eos_id)
        # Clip to context
        if len(tokens) > ctx:
            tokens = tokens[:ctx]
        if len(tokens) > 1:
            sequences.append(tokens)

    return sequences


def int8_inference(tokens: list[int], mw, tok: BPETokenizer, ctx: int) -> str:
    """Run integer forward pass, generate one response token."""
    from src.numerics import forward_int
    import numpy as np

    if len(tokens) > ctx:
        tokens = tokens[-ctx:]

    out_tokens = []
    for _ in range(20):  # max response tokens
        logits = forward_int(tokens, mw)
        next_tok = int(np.argmax(logits))
        sep_id = tok.vocab.get("<SEP>", 1)
        eos_id = tok.vocab.get("<EOS>", 3)
        if next_tok in (sep_id, eos_id):
            break
        out_tokens.append(next_tok)
        tokens = tokens + [next_tok]
        if len(tokens) > ctx:
            tokens = tokens[-ctx:]

    return tok.decode(out_tokens)


def train(corpus_path: str | None, epochs: int, output_dir: str):
    if not HAS_TORCH:
        print("PyTorch required for training. Install: pip install torch")
        sys.exit(1)

    # Config
    VOCAB = 128; EMBED = 32; N_LAYERS = 2; N_HEADS = 4; FFN = 64; CTX = 20

    # Load corpus
    if corpus_path and Path(corpus_path).exists():
        text = Path(corpus_path).read_text()
        print(f"Corpus: {corpus_path}  ({len(text)} chars)")
    else:
        text = DEFAULT_CORPUS
        print("Using built-in corpus.")

    # Train tokenizer
    print("Training BPE tokenizer...")
    tok = BPETokenizer()
    tok.train(text, VOCAB)
    print(f"  Vocab size: {len(tok.vocab)}")

    # Parse sequences
    sequences = parse_corpus(text, tok, CTX)
    print(f"  Training sequences: {len(sequences)}")
    if not sequences:
        print("No sequences found. Check corpus format: <SEP>input<SEP>response<SEP>")
        sys.exit(1)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SoulModel(VOCAB, EMBED, N_LAYERS, N_HEADS, FFN, CTX).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  (~{n_params//1000}k)")

    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)

    # Build training batches
    xs, ys = [], []
    for seq in sequences:
        for i in range(len(seq) - 1):
            ctx_tokens = seq[max(0, i - CTX + 1): i + 1]
            pad_len = CTX - len(ctx_tokens)
            ctx_padded = [0] * pad_len + ctx_tokens
            xs.append(ctx_padded)
            ys.append(seq[i + 1])

    if not xs:
        print("No training pairs generated.")
        sys.exit(1)

    X = torch.tensor(xs, dtype=torch.long, device=device)
    Y = torch.tensor(ys, dtype=torch.long, device=device)
    print(f"Training pairs: {len(xs)}")

    # Checkpoints
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    best_int8_acc = -1.0
    best_epoch    = 0

    print(f"\nTraining for {epochs} epochs...")
    print("─" * 60)

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        # Random mini-batch (use all for tiny dataset)
        logits = model(X)[:, -1, :]   # (N, vocab)
        loss = F.cross_entropy(logits, Y, label_smoothing=0.15)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        if epoch % 500 == 0 or epoch == 1:
            elapsed = time.time() - t0
            # Float inference
            model.eval()
            with torch.no_grad():
                test_seq = sequences[0][:CTX // 2]
                inp = torch.tensor([test_seq], dtype=torch.long, device=device)
                fl_logits = model(inp)[0, -1]
                fl_next = int(fl_logits.argmax().item())
                fl_response = tok.decode([fl_next])

            # Int8 inference
            mw = export_weights(model, tok, VOCAB, EMBED, N_LAYERS, N_HEADS, FFN, CTX)
            int8_resp = int8_inference(test_seq, mw, tok, CTX)

            # Int8 accuracy
            correct = 0
            for seq in sequences[:min(20, len(sequences))]:
                pred = int8_inference(seq[:len(seq)//2], mw, tok, CTX)
                ref  = tok.decode(seq[len(seq)//2:])
                correct += 1 if pred.strip() == ref.strip() else 0
            int8_acc = correct / min(20, len(sequences))

            print(f"Epoch {epoch:6d}/{epochs}  loss={loss.item():.4f}  "
                  f"int8_acc={int8_acc:.2f}  elapsed={elapsed:.0f}s")
            print(f"  float:  {tok.decode(test_seq)} → {fl_response}")
            print(f"  int8:   {tok.decode(test_seq)} → {int8_resp}")

            if int8_acc > best_int8_acc:
                best_int8_acc = int8_acc
                best_epoch = epoch
                ckpt_path = ckpt_dir / f"soul_e{epoch}_acc{int8_acc:.2f}.pt"
                torch.save(model.state_dict(), ckpt_path)
                # Save best weights
                from src.soul_io import save_soul
                save_soul(str(out_path / "soul.bin"), mw, tok.to_dict())
                print(f"  ★ New best (int8_acc={int8_acc:.2f}) saved.")

    print("─" * 60)
    print(f"Training complete. Best int8_acc={best_int8_acc:.2f} at epoch {best_epoch}")
    print(f"Weights: {out_path}/soul.bin")
    print(f"Now run:  python build.py  to build the Spectrum .TAP")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SoulPlayer for ZX Spectrum")
    parser.add_argument("corpus", nargs="?", help="Path to corpus text file")
    parser.add_argument("--epochs", type=int, default=10000, help="Training epochs (default 10000)")
    parser.add_argument("--output", default="models", help="Output directory (default: models/)")
    args = parser.parse_args()

    train(args.corpus, args.epochs, args.output)
