"""
numerics.py  –  ground-truth fixed-point transformer for SoulPlayer Spectrum
Matches the integer arithmetic that will run on the Z80.

Architecture  (same as C64 original):
  vocab        128 tokens
  embed_dim    32
  layers       2
  heads        4  x  8 dims
  ffn_hidden   64
  context      20 tokens
  params       ~25 000  (int8)

Fixed-point convention
  Weights      int8,  per-tensor power-of-2 shift  (Q0.7 effectively)
  Activations  int16, Q8.8   (upper 8 = integer, lower 8 = fraction)
  Accumulator  int32 during matmul, then >> shift >> 8 to land in Q8.8

Z80 differences vs 6502:
  * Z80 HAS an 8x8->16 bit multiply  (not used here – we keep shift-add for
    compatibility with the reference, but note it for optimisation)
  * Z80 has 16-bit register pairs (BC, DE, HL, IX, IY) – matmul inner loop
    can use these for pointer arithmetic without page tricks
  * 128k model can bank-switch extra 16k RAM pages for weights
"""

import math
import json
import struct
import numpy as np


# ---------------------------------------------------------------------------
# Fixed-point helpers
# ---------------------------------------------------------------------------

Q8_8_ONE = 256          # 1.0 in Q8.8
SHIFT     = 8           # fractional bits


def to_q88(x: float) -> int:
    """Float -> Q8.8 int16, clamped."""
    v = int(round(x * Q8_8_ONE))
    return max(-32768, min(32767, v))


def from_q88(v: int) -> float:
    return v / Q8_8_ONE


def q88_mul(a: int, b: int) -> int:
    """Q8.8 × Q8.8 -> Q8.8  (shift right 8, keep int16)."""
    return (a * b) >> SHIFT


def clamp16(v: int) -> int:
    return max(-32768, min(32767, v))


# ---------------------------------------------------------------------------
# Weight quantisation helpers
# ---------------------------------------------------------------------------

def best_power2_shift(arr: np.ndarray) -> int:
    """Find s such that arr * 2^s fits in int8 with minimal error."""
    mx = np.max(np.abs(arr))
    if mx == 0:
        return 7
    # we want mx * 2^s < 128
    s = int(math.floor(math.log2(127.0 / mx)))
    return max(0, min(14, s))


def quantize_weights(arr: np.ndarray, shift: int) -> np.ndarray:
    scale = float(1 << shift)
    q = np.round(arr * scale).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_weights(q: np.ndarray, shift: int) -> np.ndarray:
    return q.astype(np.float32) / float(1 << shift)


# ---------------------------------------------------------------------------
# Matmul  (int8 weight × int16 activation -> int16 result)
# ---------------------------------------------------------------------------

def matvec_int(W_q8: np.ndarray, x_q88: np.ndarray, w_shift: int) -> np.ndarray:
    """
    W_q8    : (out, in)  int8
    x_q88   : (in,)      int16  Q8.8
    w_shift : int        power-of-2 scale for W

    Result  : (out,)     int16  Q8.8

    Integer arithmetic:
        acc = sum_i  W[o,i] * x[i]
        Each term: int8 * int16 = int24 (safe in int32)
        After loop: acc >>= w_shift   (remove weight scale)
                    acc >>= 8         (remove Q8.8 fraction)  -- combined >> (w_shift+8) >> 1
        The C64 original does an extra >>1 post-multiply; we replicate that.
    """
    out = np.zeros(W_q8.shape[0], dtype=np.int32)
    for o in range(W_q8.shape[0]):
        acc = np.int32(0)
        for i in range(W_q8.shape[1]):
            acc += np.int32(W_q8[o, i]) * np.int32(x_q88[i])
        # >>  (w_shift + 8)  then >> 1  (matches C64 build.py × 0.5 post-shift)
        out[o] = acc >> (w_shift + 8 + 1)
    return np.clip(out, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# RMSNorm  (integer version)
# ---------------------------------------------------------------------------

def rms_norm_int(x_q88: np.ndarray, g_q88: np.ndarray) -> np.ndarray:
    """
    x_q88 : (d,)  int16  Q8.8
    g_q88 : (d,)  int16  Q8.8  (learned scale)
    returns (d,)  int16  Q8.8
    """
    d = len(x_q88)
    # sum of squares in Q8.8 * Q8.8 = Q16.16, accumulate in int32
    ss = np.int32(0)
    for v in x_q88:
        ss += np.int32(v) * np.int32(v)
    # mean: divide by d, result still Q16.16
    mean_ss = ss // d
    # sqrt: we want 1/rms in Q8.8
    # rms^2  in Q16.16 → rms in Q8.8  via integer sqrt
    rms_q88 = int(math.isqrt(int(mean_ss)))   # isqrt of Q16.16 gives Q8.8 :contenteditable
    if rms_q88 == 0:
        rms_q88 = 1
    # x / rms: (Q8.8 << 8) / Q8.8 = Q8.8
    out = np.zeros(d, dtype=np.int16)
    for i in range(d):
        norm_i = (np.int32(x_q88[i]) << 8) // np.int32(rms_q88)
        # multiply by gain g_q88[i]  (Q8.8 * Q8.8 >> 8)
        out[i] = clamp16(int(norm_i * np.int32(g_q88[i])) >> 8)
    return out


# ---------------------------------------------------------------------------
# Softmax via lookup table  (128-entry, scores >> 14 normalisation)
#   Matches the C64's key insight: shift by 14 not 17
# ---------------------------------------------------------------------------

EXP_TABLE_SIZE = 128
_exp_lut = None


def _build_exp_lut():
    global _exp_lut
    if _exp_lut is not None:
        return
    # indices 0..127 represent score values after >>14
    # index i → exp(i / 16384 * something) – we map linearly over a useful range
    # The LUT covers scores in [0, 127] after the >>14 normalisation
    # We use the same strategy as the C64: lut[i] = round(256 * exp(i/16.0 - 4))
    # clipped to 0..255  (uint8)
    _exp_lut = np.zeros(EXP_TABLE_SIZE, dtype=np.uint8)
    for i in range(EXP_TABLE_SIZE):
        v = math.exp(i / 16.0 - 4.0) * 256.0
        _exp_lut[i] = min(255, max(0, int(round(v))))


def softmax_lut(scores_q88: np.ndarray) -> np.ndarray:
    """
    scores_q88 : (seq,)  int16  Q8.8
    returns      (seq,)  int16  Q8.8  (sum ≈ 256, i.e. 1.0 in Q8.8)

    Strategy:
      1. Shift each score right by 14 to get a 0..127 index into exp LUT
      2. Look up exp values (uint8)
      3. Sum, divide each by sum → Q8.8 probability
    """
    _build_exp_lut()
    seq = len(scores_q88)
    indices = np.right_shift(scores_q88.astype(np.int32), 14 - 8)  # >>6 from Q8.8
    indices = np.clip(indices, 0, EXP_TABLE_SIZE - 1).astype(int)
    exps = np.array([int(_exp_lut[idx]) for idx in indices], dtype=np.int32)
    total = max(1, int(np.sum(exps)))
    probs = np.zeros(seq, dtype=np.int16)
    for i in range(seq):
        probs[i] = clamp16((exps[i] * Q8_8_ONE) // total)
    return probs


# ---------------------------------------------------------------------------
# Attention head
# ---------------------------------------------------------------------------

def attention_head_int(
    Q_q88: np.ndarray,   # (seq, head_dim)  int16
    K_q88: np.ndarray,   # (seq, head_dim)  int16
    V_q88: np.ndarray,   # (seq, head_dim)  int16
    seq_len: int,
) -> np.ndarray:          # (seq, head_dim)  int16
    """Causal self-attention for one head, full integer arithmetic."""
    head_dim = Q_q88.shape[1]
    out = np.zeros((seq_len, head_dim), dtype=np.int16)

    for pos in range(seq_len):
        # scores: Q[pos] · K[0..pos]  →  int32 sum of int16*int16
        scores = np.zeros(seq_len, dtype=np.int16)
        for t in range(pos + 1):
            acc = np.int32(0)
            for d in range(head_dim):
                acc += np.int32(Q_q88[pos, d]) * np.int32(K_q88[t, d])
            # normalise: >> 14 shift  (key C64 insight ported to Z80)
            scores[t] = clamp16(int(acc >> 14))

        # softmax
        probs = softmax_lut(scores[:pos + 1])

        # weighted sum of V
        for d in range(head_dim):
            acc = np.int32(0)
            for t in range(pos + 1):
                acc += np.int32(probs[t]) * np.int32(V_q88[t, d])
            out[pos, d] = clamp16(int(acc >> 8))

    return out


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class LayerWeights:
    def __init__(self):
        # RMSNorm gains
        self.rn1_g = None   # (embed,)  int16 Q8.8
        self.rn2_g = None   # (embed,)  int16 Q8.8
        # Attention projections  (head_dim = embed // n_heads)
        self.wq = None      # (embed, embed) int8
        self.wq_sh = 0
        self.wk = None
        self.wk_sh = 0
        self.wv = None
        self.wv_sh = 0
        self.wo = None      # (embed, embed) int8
        self.wo_sh = 0
        # FFN
        self.w1 = None      # (ffn, embed)  int8
        self.w1_sh = 0
        self.w2 = None      # (embed, ffn)  int8
        self.w2_sh = 0


def layer_forward_int(
    x: np.ndarray,          # (seq, embed) int16 Q8.8
    w: LayerWeights,
    n_heads: int,
    seq_len: int,
) -> np.ndarray:
    embed = x.shape[1]
    head_dim = embed // n_heads

    # --- RMSNorm 1 ---
    xn = np.array([rms_norm_int(x[t], w.rn1_g) for t in range(seq_len)])

    # --- Multi-head attention ---
    Q_flat = np.array([matvec_int(w.wq, xn[t], w.wq_sh) for t in range(seq_len)])
    K_flat = np.array([matvec_int(w.wk, xn[t], w.wk_sh) for t in range(seq_len)])
    V_flat = np.array([matvec_int(w.wv, xn[t], w.wv_sh) for t in range(seq_len)])

    # reshape for multi-head: (seq, n_heads, head_dim)
    Q = Q_flat.reshape(seq_len, n_heads, head_dim)
    K = K_flat.reshape(seq_len, n_heads, head_dim)
    V = V_flat.reshape(seq_len, n_heads, head_dim)

    head_outs = []
    for h in range(n_heads):
        head_outs.append(attention_head_int(Q[:, h], K[:, h], V[:, h], seq_len))

    attn_out = np.concatenate(head_outs, axis=1)   # (seq, embed) int16
    # output projection
    attn_proj = np.array([matvec_int(w.wo, attn_out[t], w.wo_sh) for t in range(seq_len)])

    # residual
    x = np.clip(x.astype(np.int32) + attn_proj.astype(np.int32), -32768, 32767).astype(np.int16)

    # --- RMSNorm 2 ---
    xn = np.array([rms_norm_int(x[t], w.rn2_g) for t in range(seq_len)])

    # --- FFN: ReLU MLP ---
    h1 = np.array([matvec_int(w.w1, xn[t], w.w1_sh) for t in range(seq_len)])
    h1 = np.clip(h1, 0, 32767).astype(np.int16)  # ReLU
    h2 = np.array([matvec_int(w.w2, h1[t], w.w2_sh) for t in range(seq_len)])

    # residual
    x = np.clip(x.astype(np.int32) + h2.astype(np.int32), -32768, 32767).astype(np.int16)
    return x


# ---------------------------------------------------------------------------
# Full forward pass
# ---------------------------------------------------------------------------

class ModelWeights:
    def __init__(self):
        self.embed_table = None   # (vocab, embed) int8
        self.embed_sh    = 0
        self.layers: list[LayerWeights] = []
        self.rn_final_g  = None   # (embed,) int16 Q8.8
        self.out_proj    = None   # (vocab, embed) int8
        self.out_sh      = 0

        # config
        self.vocab     = 128
        self.embed     = 32
        self.n_heads   = 4
        self.head_dim  = 8
        self.ffn       = 64
        self.ctx       = 20
        self.n_layers  = 2


def forward_int(tokens: list[int], mw: ModelWeights) -> list[int]:
    """
    Full integer forward pass.
    tokens: list of token ids (up to ctx len)
    returns: list of logits (int16 Q8.8) of length vocab
    """
    seq_len = len(tokens)
    embed = mw.embed

    # Embed
    x = np.zeros((seq_len, embed), dtype=np.int16)
    for t, tok in enumerate(tokens):
        row = mw.embed_table[tok].astype(np.int32)
        # dequant to Q8.8: left-shift by (8 - embed_sh) if embed_sh < 8 else right
        shift = 8 - mw.embed_sh
        if shift >= 0:
            x[t] = np.clip(row << shift, -32768, 32767).astype(np.int16)
        else:
            x[t] = np.clip(row >> (-shift), -32768, 32767).astype(np.int16)

    # Layers
    for layer_w in mw.layers:
        x = layer_forward_int(x, layer_w, mw.n_heads, seq_len)

    # Final RMSNorm
    x = np.array([rms_norm_int(x[t], mw.rn_final_g) for t in range(seq_len)])

    # Output projection (only last position)
    logits = matvec_int(mw.out_proj, x[-1], mw.out_sh)
    return logits.tolist()
