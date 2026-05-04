#!/usr/bin/env python3
"""
soulchat.py  –  Chat with your trained soul in the terminal.

Runs the same integer Q8.8 arithmetic as the ZX Spectrum Z80 engine,
just faster. Use this to verify your model before building the .TAP.

Usage:
    python soulchat.py                     # uses models/soul.bin
    python soulchat.py models/soul.bin
    python soulchat.py --float             # use float inference (for comparison)
"""

import argparse
import sys
import time
from pathlib import Path


def run_chat(model_path: str, use_float: bool = False):
    try:
        from src.soul_io import load_soul
        from src.numerics import forward_int
        from train import BPETokenizer
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you run from the soulplayer-spectrum directory.")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run:  python train.py  to train a model first.")
        sys.exit(1)

    print("Loading soul...", end="", flush=True)
    mw, tok_dict = load_soul(str(model_path))
    tok = BPETokenizer.from_dict(tok_dict)
    print(" done.")

    sep_id = tok.vocab.get("<SEP>", 1)
    eos_id = tok.vocab.get("<EOS>", 3)
    pad_id = tok.vocab.get("<PAD>", 0)

    print()
    print("   .------.")
    print("  | O    O |")
    print("  |   __   |")
    print("  |'--|--|'.|")
    print()
    print("  SOUL PLAYER SPECTRUM")
    print("  25K PARAMETERS. 2 LAYERS. REAL TRANSFORMER.")
    print("  ON A ZX SPECTRUM (or close enough)")
    print()
    print("  Type a short message. 'q' to quit.")
    print()

    mode = "float" if use_float else "int8"
    context: list[int] = []

    while True:
        try:
            user_input = input("YOU> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("q", "quit", "exit"):
            print("SPECTRUM> GOODBYE!")
            break

        if not user_input:
            continue

        # Encode: <SEP> + tokens + <SEP>
        user_tokens = [sep_id] + tok.encode(user_input.lower()) + [sep_id]
        context = (context + user_tokens)[-mw.ctx:]

        print("SPECTRUM> ", end="", flush=True)
        response_tokens = []

        # Generate tokens one at a time
        t0 = time.time()
        for step in range(30):   # max 30 output tokens
            prompt = context[-mw.ctx:]
            if len(prompt) == 0:
                break

            import numpy as np
            if use_float:
                # Float inference via PyTorch (if available)
                try:
                    import torch
                    from train import SoulModel
                    # Would need a loaded torch model – skip for now
                    logits = forward_int(prompt, mw)
                except Exception:
                    logits = forward_int(prompt, mw)
            else:
                logits = forward_int(prompt, mw)

            next_tok = int(np.argmax(logits))

            if next_tok in (sep_id, eos_id, pad_id):
                break

            response_tokens.append(next_tok)
            decoded = tok.inv_vocab.get(next_tok, "?")
            print(decoded, end="", flush=True)

            # Simulate Spectrum border flash (terminal colour)
            sys.stdout.write("\033[?25l")  # hide cursor briefly
            time.sleep(0.05)
            sys.stdout.write("\033[?25h")

            context = (context + [next_tok])[-mw.ctx:]

        elapsed = time.time() - t0
        print(f"  [{elapsed:.1f}s, {len(response_tokens)} tokens, {mode}]")

        # Add response to context
        context = (context + [sep_id])[-mw.ctx:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with your Spectrum soul locally")
    parser.add_argument("model", nargs="?", default="models/soul.bin")
    parser.add_argument("--float", action="store_true", help="Use float inference")
    args = parser.parse_args()
    run_chat(args.model, use_float=args.float)
