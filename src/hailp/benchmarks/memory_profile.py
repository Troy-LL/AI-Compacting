"""High-level memory profile: RAM vs sequence length for a single model.

This is the "hero" benchmark described in the H(AI)LP spec:

- X-axis: sequence length
- Y-axis: peak RAM delta during a forward pass (MB)

Compared to `memory_benchmark.py` (which focuses on KV cache vs fixed
state in bytes), this script uses the more user-facing `ram_at_seq`
utility from `training.trainer` to measure end-to-end process memory.

Run (H(AI)LP only by default; optional Baseline GPT):

    python benchmarks/memory_profile.py
    python benchmarks/memory_profile.py --baseline

The core API is ``benchmark_memory_vs_sequence_length`` which you can
reuse from notebooks or other scripts.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

# Ensure src is in sys.path so we can import 'hailp'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hailp.models.baseline_gpt import BaselineConfig, BaselineGPT
from hailp.models.hailp_model import HAILPConfig, HAILPModel
from hailp.training.device import DEVICE as DEFAULT_DEVICE
from hailp.training.device import DEVICE_NAME
from hailp.training.trainer import ram_at_seq, ram_mb


@dataclass
class MemoryPoint:
    model_name: str
    seq_len: int
    peak_ram_mb: float


def benchmark_memory_vs_sequence_length(
    model: torch.nn.Module,
    model_name: str,
    *,
    sequence_lengths: Iterable[int] | None = None,
    device: torch.device | None = None,
    is_recurrent: bool = False,
    batch_size: int = 1,
    vocab_size: int = 8_000,
) -> list[MemoryPoint]:
    """Measure peak RAM delta (MB) for a model across sequence lengths.

    Parameters
    ----------
    model:
        Model instance (BaselineGPT or H(AI)LP).
    model_name:
        Label used in the returned results table.
    sequence_lengths:
        Iterable of sequence lengths to test.  Defaults to
        ``[64, 128, 256, 512, 1024, 2048, 4096]``.
    device:
        Torch device; defaults to CUDA if available else CPU.
    is_recurrent:
        True for H(AI)LP so the helper uses the recurrent path.
    batch_size:
        Batch size for each measurement (default 1).
    vocab_size:
        Vocabulary size used when sampling random tokens.

    Returns
    -------
    List[MemoryPoint]
        One entry per sequence length.
    """
    if sequence_lengths is None:
        sequence_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    if device is None:
        device = DEFAULT_DEVICE

    model.to(device).eval()

    results: list[MemoryPoint] = []
    for seq in sequence_lengths:
        peak_mb = ram_at_seq(
            model,
            seq_len=seq,
            batch_size=batch_size,
            device=device,
            is_recurrent=is_recurrent,
            vocab_size=vocab_size,
        )
        results.append(MemoryPoint(model_name=model_name, seq_len=seq, peak_ram_mb=peak_mb))
    return results


def _print_table(points: list[MemoryPoint]) -> None:
    """Pretty print a dual-model table grouped by seq_len."""
    # Group by seq_len, preserving order
    by_seq: dict[int, list[MemoryPoint]] = {}
    order: list[int] = []
    for p in points:
        if p.seq_len not in by_seq:
            by_seq[p.seq_len] = []
            order.append(p.seq_len)
        by_seq[p.seq_len].append(p)

    header = f"{'seq_len':>8} | {'model':<15} | {'peak_ram_mb':>12}"
    print(header)
    print("-" * len(header))

    for seq in order:
        for p in by_seq[seq]:
            print(f"{p.seq_len:>8} | {p.model_name:<15} | {p.peak_ram_mb:>12.1f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="H(AI)LP memory profile (RAM delta vs sequence length); optional Baseline GPT"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also compute Baseline GPT memory points.",
    )
    args = parser.parse_args()

    device = DEFAULT_DEVICE
    print("=" * 80)
    print("H(AI)LP Memory Profile: RAM vs sequence length (forward pass)")
    print("=" * 80)
    print(f"  Runs on: {DEVICE_NAME}  ({device})")
    print()

    # Match configs used in other benchmarks (8k vocab for compactness).
    hailp_cfg = HAILPConfig(
        layers=12,
        hidden_dim=512,
        vocab_size=8000,
        ffn_sharing_group_size=4,
        low_rank_dim=64,
        adapter_rank=32,
        dropout=0.0,
    )

    hailp = HAILPModel(hailp_cfg)

    seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]

    hailp_points = benchmark_memory_vs_sequence_length(
        hailp,
        "H(AI)LP RWKV",
        sequence_lengths=seq_lens,
        device=device,
        is_recurrent=True,
        batch_size=1,
        vocab_size=hailp_cfg.vocab_size,
    )

    all_points = hailp_points[:]
    if args.baseline:
        from models.baseline_gpt import BaselineConfig, BaselineGPT

        baseline_cfg = BaselineConfig(
            layers=6,
            hidden_dim=512,
            attention_heads=8,
            ffn_expansion=4,
            vocab_size=8000,
            context_window=4096,
            dropout=0.0,
        )
        baseline = BaselineGPT(baseline_cfg)

        baseline_points = benchmark_memory_vs_sequence_length(
            baseline,
            "Baseline GPT",
            sequence_lengths=seq_lens,
            device=device,
            is_recurrent=False,
            batch_size=1,
            vocab_size=baseline_cfg.vocab_size,
        )
        all_points = baseline_points + hailp_points

    _print_table(all_points)

    print()
    print("Key intuition:")
    if args.baseline:
        print("  - Baseline GPT: RAM should grow roughly linearly with sequence length.")
    print("  - H(AI)LP RWKV: additional RAM per token is close to flat (fixed state).")
    print("Use this table as the numeric backbone for your hero RAM vs seq-length graph.")


if __name__ == "__main__":
    main()

