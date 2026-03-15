"""Memory benchmark: Baseline GPT (growing KV cache) vs LifeLink RWKV (fixed state).

Run:
    python benchmarks/memory_benchmark.py

Output columns:
    seq_len  | model        | state_bytes | kv_cache_bytes | total_bytes | growth_factor
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from models.baseline_gpt import BaselineConfig, BaselineGPT
from models.lifelink_rwkv import LifeLinkConfig, LifeLinkRWKV

# ── Config ─────────────────────────────────────────────────────────────────────

BASELINE_CFG = BaselineConfig(
    layers=6,
    hidden_dim=512,
    attention_heads=8,
    ffn_expansion=4,
    vocab_size=8000,
    context_window=512,
    dropout=0.0,
)

LIFELINK_CFG = LifeLinkConfig(
    layers=12,
    hidden_dim=512,
    vocab_size=8000,
    ffn_sharing_group_size=4,
    low_rank_dim=64,
    adapter_rank=32,
    dropout=0.0,
)

SEQUENCE_LENGTHS = [8, 16, 32, 64, 128, 256, 512]
BATCH_SIZE = 1

# ── Benchmark ──────────────────────────────────────────────────────────────────

def run_baseline_benchmark(seq_lens: list[int]) -> list[dict]:
    """Measure baseline KV cache growth."""
    model = BaselineGPT(BASELINE_CFG)
    model.eval()
    results = []

    for seq in seq_lens:
        model.clear_cache()
        x = torch.randint(0, BASELINE_CFG.vocab_size, (BATCH_SIZE, seq))
        with torch.no_grad():
            model(x, use_cache=True)
        cache_bytes = model.total_kv_cache_bytes
        results.append({
            "seq_len": seq,
            "model": "Baseline GPT",
            "state_bytes": 0,
            "kv_cache_bytes": cache_bytes,
            "total_memory_bytes": cache_bytes,
        })

    return results


def run_lifelink_benchmark(seq_lens: list[int]) -> list[dict]:
    """Measure LifeLink fixed state."""
    model = LifeLinkRWKV(LIFELINK_CFG)
    model.eval()
    results = []

    for seq in seq_lens:
        x = torch.randint(0, LIFELINK_CFG.vocab_size, (BATCH_SIZE, seq))
        with torch.no_grad():
            _, h_states = model(x)
        state_bytes = sum(h.numel() * h.element_size() for h in h_states)
        results.append({
            "seq_len": seq,
            "model": "LifeLink RWKV",
            "state_bytes": state_bytes,
            "kv_cache_bytes": 0,
            "total_memory_bytes": state_bytes,
        })

    return results


def print_table(rows: list[dict]) -> None:
    """Pretty-print a markdown-style table."""
    header = (
        f"{'seq_len':>8} | {'model':<15} | "
        f"{'kv_cache_bytes':>15} | {'state_bytes':>12} | {'total_MB':>8} | {'vs_seq8':>10}"
    )
    print(header)
    print("-" * len(header))

    baseline_at_8 = None
    lifelink_at_8 = None

    for row in rows:
        total = row["total_memory_bytes"]
        total_mb = total / 1024 / 1024

        if row["model"] == "Baseline GPT":
            if baseline_at_8 is None:
                baseline_at_8 = total
            factor = total / baseline_at_8 if baseline_at_8 else 1.0
        else:
            if lifelink_at_8 is None:
                lifelink_at_8 = total
            factor = total / lifelink_at_8 if lifelink_at_8 else 1.0

        print(
            f"{row['seq_len']:>8} | {row['model']:<15} | "
            f"{row['kv_cache_bytes']:>15,} | {row['state_bytes']:>12,} | "
            f"{total_mb:>8.3f} | {factor:>10.2f}×"
        )


def main() -> None:
    print("=" * 80)
    print("H(AI)LP Memory Benchmark: KV Cache Growth vs Fixed Recurrent State")
    print("=" * 80)
    print()

    baseline_results = run_baseline_benchmark(SEQUENCE_LENGTHS)
    lifelink_results = run_lifelink_benchmark(SEQUENCE_LENGTHS)

    # Interleave results for easy comparison
    combined = []
    for b, l in zip(baseline_results, lifelink_results):
        combined.append(b)
        combined.append(l)

    print_table(combined)
    print()
    print("Key result:")
    print("  Baseline KV cache grows linearly with sequence length.")
    print("  LifeLink RWKV state is CONSTANT regardless of sequence length.")
    print()

    # Validation assertion — the whole point
    lifelink_bytes = [r["total_memory_bytes"] for r in lifelink_results]
    assert all(b == lifelink_bytes[0] for b in lifelink_bytes), (
        "❌ FAIL: LifeLink state is NOT constant across sequence lengths!"
    )
    print("✓ PASS: LifeLink state is constant across all sequence lengths.")

    baseline_bytes = [r["total_memory_bytes"] for r in baseline_results]
    assert baseline_bytes[-1] > baseline_bytes[0], (
        "❌ FAIL: Baseline KV cache did not grow!"
    )
    print("✓ PASS: Baseline KV cache grows with sequence length.")


if __name__ == "__main__":
    main()
