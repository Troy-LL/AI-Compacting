"""demo.py — H(AI)LP project live demonstration script.

Shows the core result in ~ 60 seconds on CPU:
  1. Both models accept tokens → output logits (correct shape)
  2. H(AI)LP recurrent state is CONSTANT across sequence lengths
  3. Baseline KV cache GROWS with sequence length
  4. H(AI)LP uses 2× more layers but same parameter budget via sharing

Run:
    python demo.py

Expected output (abbreviated):
──────────────────────────────────────────────────────────────
H(AI)LP: Demonstrating Fixed-Memory LLM Architecture
──────────────────────────────────────────────────────────────

[1] Forward pass shapes
  Baseline GPT  : input (2, 64) → logits (2, 64, 8000) ✓
  H(AI)LP RWKV  : input (2, 64) → logits (2, 64, 8000) ✓

[2] Memory growth comparison
  Seq   64 | Baseline KV cache:    786,432 bytes | H(AI)LP state: 24,576 bytes
  Seq  128 | Baseline KV cache:  1,572,864 bytes | H(AI)LP state: 24,576 bytes  ← SAME
  Seq  256 | Baseline KV cache:  3,145,728 bytes | H(AI)LP state: 24,576 bytes  ← SAME
  Seq  512 | Baseline KV cache:  6,291,456 bytes | H(AI)LP state: 24,576 bytes  ← SAME

[3] Recurrent state: continuous context
  Token 1 → state[0] norm: 0.xxxx
  Token 2 → state[0] norm: 0.xxxx  (state updated!)
  State shape CONSTANT: (2, 512)

[4] Parameter counts
  Baseline GPT  : xx,xxx,xxx params
  H(AI)LP RWKV  : xx,xxx,xxx params  (shared FFNs = fewer unique params)

✓ All H(AI)LP properties demonstrated.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch

from models.baseline_gpt import BaselineConfig, BaselineGPT
from models.hailp_model import HAILPConfig, HAILPModel


# ── Config ─────────────────────────────────────────────────────────────────────
# Use full configs so parameter counts are meaningful

BASELINE_CFG = BaselineConfig(
    layers=6,
    hidden_dim=512,
    attention_heads=8,
    ffn_expansion=4,
    vocab_size=8000,
    context_window=512,
    dropout=0.0,
)

HAILP_CFG = HAILPConfig(
    layers=12,
    hidden_dim=512,
    vocab_size=8000,
    ffn_sharing_group_size=4,
    low_rank_dim=64,
    adapter_rank=32,
    dropout=0.0,
)

SEQ_LENS = [64, 128, 256, 512]
BATCH = 2


def divider(char: str = "─", width: int = 70) -> None:
    print(char * width)


def section(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


def run_demo() -> None:
    print()
    divider("═")
    print("H(AI)LP: Demonstrating Fixed-Memory LLM Architecture")
    divider("═")

    print("\nLoading models (this may take a few seconds)...")
    baseline = BaselineGPT(BASELINE_CFG)
    baseline.eval()

    hailp = HAILPModel(HAILP_CFG)
    hailp.eval()
    print("  Models loaded.")

    # ── [1] Forward pass shapes ────────────────────────────────────────────────
    section(1, "Forward pass shapes")

    tokens = torch.randint(0, BASELINE_CFG.vocab_size, (BATCH, 64))

    with torch.no_grad():
        b_logits = baseline(tokens)
    print(f"  Baseline GPT  : input {tuple(tokens.shape)} → logits {tuple(b_logits.shape)} ✓")
    assert b_logits.shape == (BATCH, 64, BASELINE_CFG.vocab_size)

    with torch.no_grad():
        ll_logits, h_states = hailp(tokens)
    print(f"  H(AI)LP RWKV  : input {tuple(tokens.shape)} → logits {tuple(ll_logits.shape)} ✓")
    assert ll_logits.shape == (BATCH, 64, HAILP_CFG.vocab_size)

    # ── [2] Memory growth comparison ───────────────────────────────────────────
    section(2, "Memory growth comparison")
    print(f"  {'Seq':>5} | {'Baseline KV cache':>22} | {'H(AI)LP state':>16}")
    divider()

    hailp_state_bytes = HAILP_CFG.state_bytes  # fixed — compute once

    for seq in SEQ_LENS:
        x = torch.randint(0, BASELINE_CFG.vocab_size, (BATCH, seq))

        # Baseline: run with cache, measure accumulated KV size
        baseline.clear_cache()
        with torch.no_grad():
            baseline(x, use_cache=True)
        kv_bytes = baseline.total_kv_cache_bytes

        # H(AI)LP state is CONSTANT — just report the config value
        # (actual state from a single forward pass with batch=2)
        with torch.no_grad():
            _, h = hailp(x)
        actual_state_bytes = sum(s.numel() * s.element_size() for s in h)

        suffix = "" if seq == SEQ_LENS[0] else "  ← SAME"
        print(
            f"  Seq {seq:>4} | Baseline KV cache: {kv_bytes:>15,} bytes |"
            f" H(AI)LP state: {actual_state_bytes:>8,} bytes{suffix}"
        )

    # Assert the key invariant
    state_sizes: list[int] = []
    for seq in SEQ_LENS:
        x = torch.randint(0, HAILP_CFG.vocab_size, (BATCH, seq))
        with torch.no_grad():
            _, h = hailp(x)
        state_sizes.append(sum(s.numel() * s.element_size() for s in h))

    assert all(s == state_sizes[0] for s in state_sizes), (
        f"❌ H(AI)LP state is NOT constant! Sizes: {state_sizes}"
    )
    print(f"\n  ✓ H(AI)LP state is CONSTANT across all sequence lengths: {state_sizes[0]:,} bytes")

    # ── [3] Recurrent state: continuous context ────────────────────────────────
    section(3, "Recurrent state: continuous context")

    h_states = None
    x1 = torch.randint(0, HAILP_CFG.vocab_size, (BATCH, 32))
    x2 = torch.randint(0, HAILP_CFG.vocab_size, (BATCH, 32))

    with torch.no_grad():
        _, h_states = hailp(x1, h_states=h_states)
        norm_before = h_states[0].norm().item()

        _, h_states_after = hailp(x2, h_states=h_states)
        norm_after = h_states_after[0].norm().item()

    print(f"  After chunk 1 → state[0] norm: {norm_before:.4f}")
    print(f"  After chunk 2 → state[0] norm: {norm_after:.4f}  (state updated!)")
    print(f"  State shape CONSTANT: {tuple(h_states_after[0].shape)}")

    assert h_states_after[0].shape == (BATCH, HAILP_CFG.hidden_dim), (
        "State shape changed!"
    )

    # ── [4] Parameter counts ───────────────────────────────────────────────────
    section(4, "Parameter counts")

    b_params = baseline.num_parameters()
    ll_params = hailp.num_parameters()

    print(f"  Baseline GPT  : {b_params:>12,} unique params (6 layers, full FFN)")
    print(f"  H(AI)LP RWKV  : {ll_params:>12,} unique params (12 layers, shared FFN)")
    print(f"  Shared FFN savings: {b_params - ll_params:+,} params")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    divider("═")
    print("✓ All H(AI)LP properties demonstrated.")
    print()
    print("Key result:")
    print(f"  Baseline KV cache at seq=512:  {baseline.total_kv_cache_bytes:,} bytes (grows with seq)")
    print(f"  H(AI)LP RWKV state at seq=512: {state_sizes[-1]:,} bytes (CONSTANT)")
    ratio = baseline.total_kv_cache_bytes / max(state_sizes[-1], 1)
    print(f"  Memory ratio: {ratio:.1f}× smaller fixed state")
    divider("═")
    print()


if __name__ == "__main__":
    run_demo()
