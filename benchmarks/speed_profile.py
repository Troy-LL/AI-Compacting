"""Speed benchmark: tokens/second for H(AI)LP (optionally compare vs Baseline GPT).

Measures forward-pass throughput at various sequence lengths and batch sizes.
Useful for comparing O(n²) attention vs O(n) recurrent scaling.

Run:
    python benchmarks/speed_profile.py

Optional:
    python benchmarks/speed_profile.py --batch-size 8 --warmup 5 --repeats 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.hailp_model import HAILPConfig, HAILPModel
from training.device import DEVICE as DEFAULT_DEVICE, DEVICE_NAME

# ── Config (match memory_benchmark / training) ───────────────────────────────────

BASELINE_CFG = None  # loaded lazily when --baseline is passed

HAILP_CFG = HAILPConfig(
    layers=12,
    hidden_dim=512,
    vocab_size=8000,
    ffn_sharing_group_size=4,
    low_rank_dim=64,
    adapter_rank=32,
    dropout=0.0,
)

DEFAULT_SEQ_LENS = [64, 128, 256, 512]
DEFAULT_BATCH_SIZE = 4
DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10


def _tokens_per_second(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    warmup: int,
    repeats: int,
    is_recurrent: bool = False,
) -> float:
    """Measure forward-pass throughput: tokens per second (after warmup)."""
    model.eval()
    total_tokens = batch_size * seq_len * repeats
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    def sync_out(out: object) -> None:
        if isinstance(out, tuple):
            out = out[0]
        assert isinstance(out, torch.Tensor)
        _ = out.reshape(-1)[0].item()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            out = model(x)
            sync_out(out)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            out = model(x)
            sync_out(out)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return total_tokens / elapsed


def run_baseline_speed(
    seq_lens: list[int],
    batch_size: int,
    device: torch.device,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> list[dict]:
    """Baseline GPT forward-pass speed at each seq_len."""
    from models.baseline_gpt import BaselineConfig, BaselineGPT

    global BASELINE_CFG
    if BASELINE_CFG is None:
        BASELINE_CFG = BaselineConfig(
            layers=6,
            hidden_dim=512,
            attention_heads=8,
            ffn_expansion=4,
            vocab_size=8000,
            context_window=512,
            dropout=0.0,
        )

    model = BaselineGPT(BASELINE_CFG).to(device).eval()
    results = []
    for seq in seq_lens:
        tps = _tokens_per_second(
            model, batch_size, seq, BASELINE_CFG.vocab_size, device, warmup, repeats, False
        )
        results.append({
            "model": "Baseline GPT",
            "seq_len": seq,
            "batch_size": batch_size,
            "tokens_per_second": tps,
        })
    return results


def run_hailp_speed(
    seq_lens: list[int],
    batch_size: int,
    device: torch.device,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> list[dict]:
    """H(AI)LP RWKV forward-pass speed at each seq_len."""
    model = HAILPModel(HAILP_CFG).to(device).eval()
    results = []
    for seq in seq_lens:
        tps = _tokens_per_second(
            model, batch_size, seq, HAILP_CFG.vocab_size, device, warmup, repeats, True
        )
        results.append({
            "model": "H(AI)LP RWKV",
            "seq_len": seq,
            "batch_size": batch_size,
            "tokens_per_second": tps,
        })
    return results


def benchmark_speed_vs_sequence_length(
    model: torch.nn.Module,
    model_name: str,
    *,
    seq_lens: list[int],
    batch_size: int,
    device: torch.device,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    is_recurrent: bool = False,
    vocab_size: int = 8_000,
) -> list[dict]:
    """Generic speed benchmark used by the spec.

    This mirrors ``benchmark_speed_vs_sequence_length`` in the design
    doc: measure tokens/second at different sequence lengths for an
    arbitrary model.  The CLI still uses the Baseline/H(AI)LP helpers
    above, but notebooks and higher-level scripts can call this
    function directly.
    """
    model = model.to(device).eval()
    results: list[dict] = []
    for seq in seq_lens:
        tps = _tokens_per_second(
            model,
            batch_size=batch_size,
            seq_len=seq,
            vocab_size=vocab_size,
            device=device,
            warmup=warmup,
            repeats=repeats,
            is_recurrent=is_recurrent,
        )
        results.append(
            {
                "model": model_name,
                "seq_len": seq,
                "batch_size": batch_size,
                "tokens_per_second": tps,
            }
        )
    return results


def print_table(rows: list[dict]) -> None:
    """Print a simple table of seq_len | model | tokens/sec."""
    header = f"{'seq_len':>8} | {'model':<15} | {'batch':>6} | {'tokens/sec':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['seq_len']:>8} | {r['model']:<15} | {r['batch_size']:>6} | "
            f"{r['tokens_per_second']:>12,.0f}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="H(AI)LP speed benchmark: tokens/sec",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for forward pass",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENS,
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup forward passes per (model, seq_len)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Repeated forward passes for timing",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also compute Baseline GPT speed.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="Device to run on",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = DEFAULT_DEVICE
        print(f"Detected device: {DEVICE_NAME} ({device})")
    else:
        device = torch.device(args.device)
    print("=" * 60)
    print("H(AI)LP Speed Benchmark")
    print("=" * 60)
    print(f"  Device: {device}  |  batch_size={args.batch_size}")
    print(
        f"  seq_lens: {args.seq_lens}  |  warmup={args.warmup}  repeats={args.repeats}"
    )
    print()

    hailp_results = run_hailp_speed(
        args.seq_lens, args.batch_size, device, args.warmup, args.repeats
    )

    combined = hailp_results[:]
    if args.baseline:
        baseline_results = run_baseline_speed(
            args.seq_lens, args.batch_size, device, args.warmup, args.repeats
        )
        combined = []
        for b, h in zip(baseline_results, hailp_results):
            combined.append(b)
            combined.append(h)

    print_table(combined)
    print()
    if args.baseline:
        print("  Baseline uses O(n²) attention; H(AI)LP uses O(n) time-mixing.")
        print("  At long sequences, H(AI)LP can show better scaling.")


if __name__ == "__main__":
    main()
