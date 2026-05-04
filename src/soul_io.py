"""
soul_io.py  –  .bin weight file writer/reader for SoulPlayer Spectrum

File format  (little-endian throughout, compatible with original C64 soul.bin)
─────────────────────────────────────────────────────────────────────────────
Offset  Size  Description
0       4     Magic  "SOUL"
4       2     Version  0x0002  (Spectrum port)
6       2     vocab      (128)
8       2     embed      (32)
10      2     n_layers   (2)
12      2     n_heads    (4)
14      2     ffn        (64)
16      2     ctx        (20)
18      2     reserved   0x0000
─────────────────────────────────────────────────────────────────────────────
Then for each weight tensor, a tensor header followed by raw bytes:
  1 byte   tensor_id   (enum below)
  1 byte   shift       (power-of-2 scale; 0xFF = Q8.8 int16)
  2 bytes  rows
  2 bytes  cols        (1 for vectors)
  rows*cols bytes  (int8) or rows*cols*2 bytes (int16 if shift==0xFF)

Tensor IDs
  0x01  embed_table   (vocab × embed)  int8
  0x10+layer*0x10  per-layer (layer 0 = 0x10, layer 1 = 0x20):
    +0x01  rn1_g   (embed,)  int16
    +0x02  rn2_g   (embed,)  int16
    +0x03  wq      (embed × embed)  int8
    +0x04  wk      (embed × embed)  int8
    +0x05  wv      (embed × embed)  int8
    +0x06  wo      (embed × embed)  int8
    +0x07  w1      (ffn × embed)    int8
    +0x08  w2      (embed × ffn)    int8
  0x80  rn_final_g   (embed,)  int16
  0x81  out_proj     (vocab × embed)  int8

Total weight data ≈ 25 KB.
"""

import struct
import numpy as np
from pathlib import Path
from src.numerics import ModelWeights, LayerWeights, to_q88, Q8_8_ONE


MAGIC   = b"SOUL"
VERSION = 0x0002


def save_soul(path: str, mw: ModelWeights, tokenizer: dict):
    """Write soul.bin + tokenizer.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        # header
        f.write(MAGIC)
        f.write(struct.pack("<HHHHHHHH",
            VERSION, mw.vocab, mw.embed, mw.n_layers,
            mw.n_heads, mw.ffn, mw.ctx, 0))

        def write_int8_tensor(tid, arr2d, shift):
            rows, cols = arr2d.shape if arr2d.ndim == 2 else (arr2d.shape[0], 1)
            f.write(struct.pack("<BBHHb", tid, shift & 0xFF, rows, cols, 0))
            # fix struct: use proper packing
            f.write(struct.pack(f"B B H H", tid & 0xFF, shift & 0xFF, rows, cols))
            # (rewrite cleanly)
            pass

        def w8(tid: int, arr: np.ndarray, shift: int):
            """Write int8 tensor."""
            flat = arr.flatten()
            rows = arr.shape[0]
            cols = arr.shape[1] if arr.ndim > 1 else 1
            header = struct.pack("<BBHH", tid & 0xFF, shift & 0xFF, rows, cols)
            f.write(header)
            f.write(flat.astype(np.int8).tobytes())

        def w16(tid: int, arr: np.ndarray):
            """Write int16 Q8.8 tensor (shift=0xFF marker)."""
            flat = arr.flatten()
            rows = arr.shape[0]
            cols = arr.shape[1] if arr.ndim > 1 else 1
            header = struct.pack("<BBHH", tid & 0xFF, 0xFF, rows, cols)
            f.write(header)
            f.write(flat.astype(np.int16).tobytes())

        # embed table
        w8(0x01, mw.embed_table, mw.embed_sh)

        # layers
        for li, lw in enumerate(mw.layers):
            base = 0x10 + li * 0x10
            w16(base + 0x01, lw.rn1_g)
            w16(base + 0x02, lw.rn2_g)
            w8(base  + 0x03, lw.wq, lw.wq_sh)
            w8(base  + 0x04, lw.wk, lw.wk_sh)
            w8(base  + 0x05, lw.wv, lw.wv_sh)
            w8(base  + 0x06, lw.wo, lw.wo_sh)
            w8(base  + 0x07, lw.w1, lw.w1_sh)
            w8(base  + 0x08, lw.w2, lw.w2_sh)

        # final norm + output
        w16(0x80, mw.rn_final_g)
        w8(0x81, mw.out_proj, mw.out_sh)

    # tokenizer
    import json
    tok_path = path.parent / "tokenizer.json"
    with open(tok_path, "w") as f:
        json.dump(tokenizer, f, indent=2)

    print(f"Saved {path}  ({path.stat().st_size} bytes)")
    print(f"Saved {tok_path}")


def load_soul(path: str):
    """Load soul.bin → (ModelWeights, tokenizer_dict)."""
    import json
    path = Path(path)

    mw = ModelWeights()
    tok_path = path.parent / "tokenizer.json"
    with open(tok_path) as f:
        tokenizer = json.load(f)

    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        ver, vocab, embed, n_layers, n_heads, ffn, ctx, _ = struct.unpack("<HHHHHHHH", f.read(16))
        mw.vocab = vocab; mw.embed = embed; mw.n_layers = n_layers
        mw.n_heads = n_heads; mw.ffn = ffn; mw.ctx = ctx
        mw.head_dim = embed // n_heads
        mw.layers = [LayerWeights() for _ in range(n_layers)]

        def read_tensor(f):
            hdr = f.read(6)
            if len(hdr) < 6:
                return None, None, None, None
            tid, shift, rows, cols = struct.unpack("<BBHH", hdr)
            count = rows * cols
            if shift == 0xFF:
                data = np.frombuffer(f.read(count * 2), dtype=np.int16)
            else:
                data = np.frombuffer(f.read(count), dtype=np.int8)
            return tid, shift, data.reshape(rows, cols) if cols > 1 else data.reshape(rows), shift

        while True:
            tid, shift, arr, _ = read_tensor(f)
            if tid is None:
                break
            if tid == 0x01:
                mw.embed_table = arr; mw.embed_sh = shift
            elif tid == 0x80:
                mw.rn_final_g = arr
            elif tid == 0x81:
                mw.out_proj = arr; mw.out_sh = shift
            else:
                li = (tid >> 4) - 1
                sub = tid & 0x0F
                if 0 <= li < n_layers:
                    lw = mw.layers[li]
                    if sub == 0x01: lw.rn1_g = arr
                    elif sub == 0x02: lw.rn2_g = arr
                    elif sub == 0x03: lw.wq = arr; lw.wq_sh = shift
                    elif sub == 0x04: lw.wk = arr; lw.wk_sh = shift
                    elif sub == 0x05: lw.wv = arr; lw.wv_sh = shift
                    elif sub == 0x06: lw.wo = arr; lw.wo_sh = shift
                    elif sub == 0x07: lw.w1 = arr; lw.w1_sh = shift
                    elif sub == 0x08: lw.w2 = arr; lw.w2_sh = shift

    return mw, tokenizer
