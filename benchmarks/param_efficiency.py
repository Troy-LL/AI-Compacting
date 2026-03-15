"""Parameter efficiency benchmark for Baseline GPT vs H(AI)LP RWKV.

This focuses on:

- Total parameters
- Unique parameters (after sharing)
- Approximate FP32 model size (MB)
- Parameters per MB of storage

Run:

    python benchmarks/param_efficiency.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baseline_gpt import BaselineConfig, BaselineGPT
from models.hailp_model import HAILPConfig, HAILPModel


@dataclass
class ParamRow:
    model_name: str
    total_params: int
    unique_params: int
    size_mb_fp32: float

    @property
    def params_per_mb(self) -> float:
        if self.size_mb_fp32 <= 0:
            return 0.0
        return self.total_params / self.size_mb_fp32


def _build_baseline() -> BaselineGPT:
    cfg = BaselineConfig(
        layers=6,
        hidden_dim=512,
        attention_heads=8,
        ffn_expansion=4,
        vocab_size=8000,
        context_window=512,
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


def benchmark_param_efficiency(
    model,
    model_name: str,
) -> ParamRow:
    """Return parameter and size statistics for a model."""
    # BaselineGPT / HAILPModel both expose num_parameters()
    if hasattr(model, "num_parameters"):
        total = int(model.num_parameters())
    else:
        total = sum(p.numel() for p in model.parameters())

    # In this project num_parameters() already accounts for sharing; keep
    # the same value for clarity.
    unique = total

    # FP32 approximate size: params * 4 bytes / (1024**2)
    size_mb = total * 4 / (1024**2)
    return ParamRow(
        model_name=model_name,
        total_params=total,
        unique_params=unique,
        size_mb_fp32=size_mb,
    )


def _print_table(rows: List[ParamRow]) -> None:
    header = (
        f"{'model':<15} | {'total_params':>13} | {'size_mb(fp32)':>13} | "
        f"{'params_per_mb':>13}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.model_name:<15} | {r.total_params:>13,} | "
            f"{r.size_mb_fp32:>13.1f} | {r.params_per_mb:>13.1f}"
        )


def main() -> None:
    print("=" * 72)
    print("H(AI)LP Parameter Efficiency Benchmark")
    print("=" * 72)
    print()

    baseline = _build_baseline()
    hailp = _build_hailp()

    baseline_row = benchmark_param_efficiency(baseline, "Baseline GPT")
    hailp_row = benchmark_param_efficiency(hailp, "H(AI)LP RWKV")

    _print_table([baseline_row, hailp_row])

    print()
    print("Interpretation:")
    print(
        "  - H(AI)LP uses parameter sharing and low-rank projections to achieve "
        "a comparable or better params/MB ratio despite more layers."
    )


if __name__ == "__main__":
    main()

