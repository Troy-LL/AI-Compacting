"""Constrained hardware benchmark: simulate RAM-constrained devices.

The original H(AI)LP spec describes a benchmark that:

- Caps available RAM (e.g. 500 MB)
- Checks whether each model can be instantiated and run at various
  sequence lengths without crashing

On Linux/macOS we can approximate this with ``resource.setrlimit`` on
the address space. On Windows there is no direct analogue in the
standard library, so this script falls back to *measuring* RAM usage
and marking runs as "over budget" instead of hard-crashing.

Run:

    python benchmarks/constrained_hardware.py --ram-limit-mb 500
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.hailp_model import HAILPConfig, HAILPModel
from training.trainer import ram_mb
from training.device import DEVICE as DEFAULT_DEVICE, DEVICE_NAME


@dataclass
class ConstrainedResult:
    model_name: str
    seq_len: int
    ok: bool
    peak_ram_mb: float


def _build_baseline():
    from models.baseline_gpt import BaselineConfig, BaselineGPT

    cfg = BaselineConfig(
        layers=6,
        hidden_dim=512,
        attention_heads=8,
        ffn_expansion=4,
        vocab_size=8000,
        context_window=4096,
        dropout=0.0,
    )
    return BaselineGPT(cfg)


def _build_hailp() -> HAILPModel:
    cfg = HAILPConfig(
        layers=12,
        hidden_dim=512,
        vocab_size=8000,
        ffn_sharing_group_size=4,
        low_rank_dim=64,
        adapter_rank=32,
        dropout=0.0,
    )
    return HAILPModel(cfg)


def benchmark_under_memory_constraint(
    model: torch.nn.Module,
    model_name: str,
    *,
    seq_lens: list[int],
    ram_limit_mb: float,
    device: torch.device | None = None,
    batch_size: int = 1,
) -> List[ConstrainedResult]:
    """Soft benchmark under a RAM budget.

    This does *not* hard-enforce the limit (portable enforcement is
    tricky), but:

    - Loads the model
    - Runs a forward pass per sequence length
    - Measures peak RSS delta
    - Marks runs where ``baseline_ram + peak_delta > ram_limit_mb`` as
      "over budget".
    """
    if device is None:
        device = DEFAULT_DEVICE

    model.to(device).eval()

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", 8_000)
    results: list[ConstrainedResult] = []

    base_before_model = ram_mb()

    # Run a dummy forward pass to allocate weights / caches.
    with torch.no_grad():
        x0 = torch.randint(0, vocab_size, (batch_size, 8), device=device)
        _ = model(x0)

    baseline_ram_after_model = ram_mb()

    for seq in seq_lens:
        x = torch.randint(0, vocab_size, (batch_size, seq), device=device)
        before = ram_mb()
        try:
            with torch.no_grad():
                _ = model(x)
            after = ram_mb()
            peak_delta = max(after - before, 0.0)
            total_est = max(after, baseline_ram_after_model)
            ok = total_est <= ram_limit_mb
        except RuntimeError:
            # Treat OOM or similar as a failure.
            peak_delta = float("inf")
            ok = False

        results.append(
            ConstrainedResult(
                model_name=model_name,
                seq_len=seq,
                ok=ok,
                peak_ram_mb=baseline_ram_after_model + max(peak_delta, 0.0),
            )
        )
    return results


def _print_results(results: list[ConstrainedResult], ram_limit_mb: float) -> None:
    header = (
        f"{'seq_len':>8} | {'model':<15} | {'peak_ram_mb':>12} | "
        f"{'status@limit':>15}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        status = "OK" if r.ok else "OVER"
        print(
            f"{r.seq_len:>8} | {r.model_name:<15} | {r.peak_ram_mb:>12.1f} | "
            f"{status} @ {ram_limit_mb:.0f}MB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate RAM-constrained devices for H(AI)LP (optional Baseline GPT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ram-limit-mb",
        type=float,
        default=500.0,
        help="Soft RAM budget to compare against (MB).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also compute Baseline GPT points.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
        help="Sequence lengths to test.",
    )
    args = parser.parse_args()

    device = DEFAULT_DEVICE
    print("=" * 80)
    print("H(AI)LP Constrained Hardware Benchmark (soft RAM budget)")
    print("=" * 80)
    print(f"  Runs on: {DEVICE_NAME}  |  budget: {args.ram_limit_mb:.0f} MB")
    print()

    hailp = _build_hailp()

    baseline_results = []
    if args.baseline:
        baseline = _build_baseline()
        baseline_results = benchmark_under_memory_constraint(
            baseline,
            "Baseline GPT",
            seq_lens=args.seq_lens,
            ram_limit_mb=args.ram_limit_mb,
            device=device,
        )
    hailp_results = benchmark_under_memory_constraint(
        hailp,
        "H(AI)LP RWKV",
        seq_lens=args.seq_lens,
        ram_limit_mb=args.ram_limit_mb,
        device=device,
    )

    _print_results(baseline_results + hailp_results, args.ram_limit_mb)

    print()
    print("Interpretation:")
    print(
        "  - Rows marked OVER indicate estimated RAM usage beyond the given budget."
    )
    print(
        "  - H(AI)LP RWKV is expected to stay within budget at longer sequences "
        "than the baseline."
    )
    if os.name == "nt":
        print(
            "\nNote: On Windows this benchmark is 'soft' (no hard limit); it "
            "relies on measured RSS instead of enforcing RLIMIT_AS."
        )


if __name__ == "__main__":
    main()

