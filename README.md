# Soul Player Spectrum

**A real 25k-parameter transformer running on a ZX Spectrum 48k or 128k.**

```
  ████████████████
  █              █
  █  O        O  █
  █     ____     █
  █    |    |    █
  ████████████████

  SOUL PLAYER SPECTRUM

  25K PARAMETERS. 2 LAYERS. REAL TRANSFORMER.
  LOADED OFF TAPE.

  YOU> hello
  SPECTRUM> hey there! so good to hear from you.
```

A 2-layer decoder-only transformer — the same architecture behind ChatGPT, Claude, and Gemini — implemented in Z80 assembly and running on an unmodified ZX Spectrum 48k or 128k. ~25,000 int8 parameters. Real multi-head causal self-attention, real softmax, real RMSNorm. The whole thing fits on a cassette tape with room to spare.

This is a direct port of [gizmo64k's soulplayer-c64](https://github.com/gizmo64k/soulplayer-c64) from 6502/6510 to Z80 assembly, adapted for the ZX Spectrum's memory map, I/O ports, and ROM routines.

---

## Differences from the C64 version

| | C64 | Spectrum 48k | Spectrum 128k |
|---|---|---|---|
| CPU | 6510 @ 1 MHz | Z80A @ 3.5 MHz | Z80A @ 3.5 MHz |
| RAM | 64 KB | 48 KB | 128 KB |
| Multiply | shift-and-add | shift-and-add (Z80 has no MUL) | same |
| Weight storage | 25 KB in RAM | 25 KB in RAM | can use bank pages |
| I/O (thinking) | SID blip + border colour | beeper blip + border colour | same |
| I/O (output) | ROM PRINT | RST 0x10 ROM call | same |
| Storage | D64 floppy image | TAP cassette image | TAP cassette image |
| Load time | fast (1541 drive) | ~45 seconds (standard tape speed) | same |
| Inference | ~60s/token | ~17s/token (3.5MHz vs 1MHz) | ~17s/token |

The Z80 is 3.5× faster than the 6502, so expect **~17 seconds per token** instead of 60. A full response still takes 1–3 minutes.

Key Z80 adaptation notes:
- The Z80 lacks a hardware multiply, so shift-and-add is used throughout (same as 6502).
- Z80 16-bit register pairs (BC, DE, HL, IX, IY) simplify pointer arithmetic vs the 6502's page tricks.
- On 128k, extra RAM pages are bank-switched via OUT (0x7FFD) to hold overflow weight data.
- The softmax LUT trick (shift scores by 14 bits, not 17) is preserved unchanged.
- Memory map differs — see below.

---

## Architecture

Identical to the C64 original:

| | |
|---|---|
| Vocab | 128 tokens (4 special + 34 chars/punct + 90 BPE merges) |
| Embedding | 32 dimensions |
| Layers | 2 |
| Attention | 4 heads × 8 dims per head |
| FFN | 64 hidden units |
| Context | 20 tokens |
| Parameters | ~25,000 (all int8) |
| Weight size | ~25 KB |
| Decoding | Greedy (argmax) |

---

## Memory map (48k)

```
0x5B00–0x5FFF   Spectrum system variables + stack
0x6000–0x6FFF   Z80 machine code (~3.5 KB)
0x7000–0xB4FF   Weights (~25.3 KB)
0xB500–0xBFFF   Activation buffers (2.75 KB)
0xC000–0xC0FF   Token ring buffer (20 × 2 bytes)
0xC100–0xC1FF   Input line buffer
0xC200–0xC2FF   Output buffer
0xC300–0xC3FF   Scratch / variables
0xF000–0xF0FF   Exp lookup table (256 bytes)
```

Stack pointer: `0xBEFF`

## Memory map (128k additional)

```
Bank page 1 (0xC000–0xFFFF):  overflow weights if >28 KB
Bank page 2:                   reserved for future larger models
```

Bank switching via `OUT (0x7FFD), A` (bits 0-2 = page number).

---

## Activation buffer layout

All activations are int16 Q8.8 (upper 8 bits = integer, lower 8 = fraction).

```
ACT_BASE + 0x0000  x        (seq × embed = 20×32×2 = 1280 bytes)
ACT_BASE + 0x0500  x_norm   (1280 bytes, after RMSNorm)
ACT_BASE + 0x0A00  Q        (1280 bytes, query projections)
ACT_BASE + 0x0F00  K        (1280 bytes, key projections)
ACT_BASE + 0x1400  V        (1280 bytes, value projections)
ACT_BASE + 0x1900  attn_out (1280 bytes, attention output)
ACT_BASE + 0x1E00  h1       (seq × ffn × 2 = 2560 bytes, FFN hidden)
ACT_BASE + 0x2800  scores   (seq × 2 = 40 bytes, attention scores per head)
```

---

## Quick start – run the pre-built soul

1. Grab `tap/soulplayer48.tap` or `tap/soulplayer128.tap`
2. Load in any ZX Spectrum emulator:
   - [Fuse](http://fuse-emulator.sourceforge.net/) (Linux/Mac/Windows) — recommended
   - [ZXSpin](http://www.zxspin.co.uk/) (Windows)
   - [SPIN](http://www.javalynx.com/spin.htm) (cross-platform)
   - [SpecEmu](https://specemu.zxe.eu/) (Windows)
3. Select Insert Tape → open .TAP file
4. Type `LOAD ""` and press ENTER
5. Start tape playback
6. Wait ~45 seconds for loading

Then type a short message in lowercase and press ENTER. The border flashes while thinking. Each output token gets a beeper blip.

> **Tip:** Stick to lowercase letters, spaces, and punctuation (`. , ! ? ' : ; -`). Capital letters become `<UNK>` tokens.

---

## Train your own soul

### 1. Install dependencies

```bash
pip install numpy torch
```

### 2. Write a corpus

Create a text file with one exchange per line in `<SEP>input<SEP>response<SEP>` format:

```
<SEP>hello<SEP>hey! nice to see you!<SEP>
<SEP>i am sad<SEP>i hear you. i care about you.<SEP>
<SEP>tell me a joke<SEP>why did the bit flip? it was tired!<SEP>
```

Keep exchanges short — the model has a 20-token context window. See `data/example_corpus.txt` for a starter.

### 3. Train

```bash
python train.py data/example_corpus.txt
python train.py data/my_corpus.txt --epochs 30000 --output models/
python train.py    # uses built-in emotional support corpus
```

This trains a BPE tokenizer (128 tokens), runs quantization-aware training, exports `models/soul.bin` and `models/tokenizer.json`.

Every 500 epochs you'll see both **float** and **int8** inference output side by side — what the model learned vs what the Spectrum will actually produce. Best checkpoint is saved by int8 quality, not float loss.

### 4. Build the Spectrum binary

```bash
python build.py
python build.py --48only
python build.py --128only
```

Produces `tap/soulplayer48.tap` and/or `tap/soulplayer128.tap`.

### 5. Run

Load `tap/soulplayer48.tap` in Fuse or any Spectrum emulator, `LOAD ""`, enjoy.

---

## Chat locally

```bash
python soulchat.py                   # uses models/soul.bin
python soulchat.py models/soul.bin
python soulchat.py --float           # compare with float inference
```

Runs the same integer Q8.8 arithmetic as the Z80 engine, just faster. Use this to verify the model before building the tape.

---

## Fixed-point arithmetic

All the same fixed-point conventions as the C64 original:

- **Weights**: int8 with per-tensor power-of-2 shifts (QAT-trained)
- **Activations**: int16 Q8.8 (upper 8 = integer, lower 8 = fraction)
- **Accumulators**: int32 during matmul, then `>> (w_shift + 9)` to land in Q8.8
- **RMSNorm**: integer sqrt via binary search; gain in Q8.8
- **Softmax**: 128-entry exp LUT, scores normalised by `>> 14` (the key insight from C64)
- **Post-matmul scale**: `× 0.5` (extra `>> 1`) to prevent overflow, matching QAT training

The Z80 matmul inner loop:
```
; W[o,i] (int8) × x[i] (int16 Q8.8) → accumulate in int32
; BC = sign-extended W[o,i]
LD A, (IX+0)   ; load weight byte
LD C, A        ; C = weight
ADD A, A       ; sign bit → carry
SBC A, A       ; A = 0x00 or 0xFF (sign mask)
LD B, A        ; BC = sign-extended weight (int16)
INC IX
LD L, (IY+0)   ; load activation low byte
LD H, (IY+1)   ; load activation high byte
INC IY : INC IY
CALL mul16     ; BC × HL → DE:HL (signed 32-bit)
CALL add32     ; DE:HL → [acc32]
```

---

## Caveats

- **It's not smart.** 25K parameters is about 70 million times smaller than GPT-4. Broken sentences are expected. The architecture works at this scale — that's the point.
- **It's contemplative.** ~17 seconds per token at 3.5 MHz. A full response takes 1–3 minutes.
- **Capitals become `<UNK>`.** Stick to lowercase.
- **Small vocabulary.** 128 tokens, 20-token context. Keep corpus exchanges short.
- **Real hardware**: On a real Spectrum, load via a real cassette or DivMMC/Interface 1 with the .TAP converted to audio.

---

## File structure

```
soulplayer-spectrum/
├── train.py              ← train a model + export weights
├── build.py              ← assemble the Spectrum .TAP
├── soulchat.py           ← chat in your terminal
│
├── data/
│   └── example_corpus.txt
│
├── models/
│   ├── soul.bin           ← trained weights (~25 KB, int8)
│   ├── tokenizer.json     ← BPE tokenizer (128 tokens)
│   └── checkpoints/       ← training checkpoints
│
├── tap/
│   ├── soulplayer48.tap   ← ready-to-run 48k tape image
│   └── soulplayer128.tap  ← ready-to-run 128k tape image
│
└── src/
    ├── numerics.py        ← ground truth: fixed-point math + forward pass
    ├── soul_io.py         ← .bin weight file format (TAP-loadable)
    └── asm_z80.py         ← Z80 assembly routines (matvec, rms_norm, softmax, etc.)
```

---

## Credits

- **C64 original**: [gizmo64k](https://github.com/gizmo64k/soulplayer-c64) — all credit for the architecture, the softmax LUT insight, and the quantisation strategy
- **Spectrum port**: adapted Z80 assembly, Spectrum I/O, TAP format
- **Debugging assistant**: Claude (Sonnet 4.6) by Anthropic
- **Lucky soul**: The ZX Spectrum by Sinclair Research, 1982

## License

GNU General Public License v3 — same as the C64 original.

---

*Three and a half megahertz. Twenty-five thousand parameters. One soul.*
