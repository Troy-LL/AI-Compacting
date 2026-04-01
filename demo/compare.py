"""Terminal demo for H(AI)LP (optionally compare vs Baseline GPT).

Run:

    uv run python demo/compare.py

This script does NOT require trained checkpoints; it uses fresh models
to demonstrate:

- Parameter counts and approximate FP32 sizes
- Memory usage vs sequence length (RSS delta)
- Forward speed (tokens/sec) at a modest context length
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.hailp_model import HAILPConfig, HAILPModel
from training.device import DEVICE as DEFAULT_DEVICE
from training.device import DEVICE_NAME
from training.trainer import ram_at_seq


def _human_mb(params: int) -> float:
    return params * 4 / (1024**2)


def _sync_first_value(out: object) -> float:
    """Force device execution and return a scalar sample value."""
    if isinstance(out, tuple):
        out = out[0]
    assert isinstance(out, torch.Tensor)
    # `.item()` forces synchronization and prevents lazy/asynchronous dispatch
    # from skewing timing (CUDA and DirectML both benefit).
    return float(out.reshape(-1)[0].item())


def _measure_speed(
    model: torch.nn.Module,
    *,
    device: torch.device,
    seq_len: int,
    batch_size: int = 1,
    repeats: int = 5,
) -> float:
    """Return tokens/sec for a simple forward loop."""
    model = model.to(device).eval()
    vocab = getattr(getattr(model, "config", None), "vocab_size", 50_257)
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        out = model(x)
        _sync_first_value(out)

    total_tokens = batch_size * seq_len * repeats
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            out = model(x)
            _sync_first_value(out)
    dt = time.perf_counter() - t0
    return total_tokens / max(dt, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="H(AI)LP terminal demo")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also build/show Baseline GPT comparisons (slower).",
    )
    args = parser.parse_args()

    device = DEFAULT_DEVICE
    print("\nH(AI)LP Terminal Demo")
    print("=" * 70)
    print(f"Runs on: {DEVICE_NAME}  ({device})")

    h_cfg = HAILPConfig(
        vocab_size=50_257,
        layers=12,
        hidden_dim=512,
        ffn_sharing_group_size=4,
        low_rank_dim=64,
        adapter_rank=32,
        dropout=0.0,
    )

    hailp = HAILPModel(h_cfg)
    h_params = hailp.num_parameters()

    print("\n[1] Parameter counts and sizes (FP32)")
    print(f"  H(AI)LP RWKV  : {h_params:>12,} params  (~{_human_mb(h_params):5.1f} MB)")

    fixed_state_bytes = h_cfg.state_bytes
    print("\n[2] Fixed state (the key invariant)")
    print(f"  Fixed state: {fixed_state_bytes:,} bytes at any sequence length")

    if args.baseline:
        from models.baseline_gpt import BaselineConfig, BaselineGPT

        b_cfg = BaselineConfig(
            vocab_size=50_257,
            layers=6,
            attention_heads=8,
            hidden_dim=512,
            ffn_expansion=4,
            context_window=4096,
            dropout=0.0,
        )
        baseline = BaselineGPT(b_cfg)

        print("\n[2b] Memory usage vs sequence length (RSS delta, MB)")
        seq_512 = 512
        seq_2048 = 2048
        b_ram_512 = ram_at_seq(
            baseline.to(device), seq_512, device=device, is_recurrent=False
        )
        h_ram_512 = ram_at_seq(
            hailp.to(device), seq_512, device=device, is_recurrent=True
        )
        b_ram_2048 = ram_at_seq(
            baseline.to(device), seq_2048, device=device, is_recurrent=False
        )
        h_ram_2048 = ram_at_seq(
            hailp.to(device), seq_2048, device=device, is_recurrent=True
        )

        print(
            f"  At seq={seq_512}:  Baseline ~{b_ram_512:6.1f} MB   |   H(AI)LP ~{h_ram_512:6.1f} MB"
        )
        print(
            f"  At seq={seq_2048}: Baseline ~{b_ram_2048:6.1f} MB   |   H(AI)LP ~{h_ram_2048:6.1f} MB"
        )

    print("\n[3] Forward speed (tokens/sec, untrained models)")
    tps_hailp = _measure_speed(
        HAILPModel(h_cfg), device=device, seq_len=512, batch_size=1, repeats=5
    )
    print(f"  H(AI)LP RWKV  : {tps_hailp:8.1f} tok/s at seq=512")
    if args.baseline:
        tps_baseline = _measure_speed(
            BaselineGPT(b_cfg), device=device, seq_len=512, batch_size=1, repeats=5
        )
        print(f"  Baseline GPT  : {tps_baseline:8.1f} tok/s at seq=512")

    # Analytic KV cache ratio (based on the baseline config in this repo).
    # Baseline KV cache grows as: 2 × layers × heads × seq × head_dim × bytes_per_elem.
    # For this repo's baseline config: head_dim=64 and bytes_per_elem=4.
    kv_bytes_seq512 = 12_582_912
    ratio = kv_bytes_seq512 / max(fixed_state_bytes, 1)
    print("\n[4] KV-cache size ratio (analytical)")
    print(f"  At seq=512: baseline KV cache is ~{ratio:.0f}x larger than H(AI)LP fixed state")
    print("  References:")
    print("   - GPT-2: https://arxiv.org/abs/1706.03762")
    print("   - llama.cpp: https://github.com/ggerganov/llama.cpp")

    print("\nNote: these numbers are from randomly initialised models on this machine.")
    print("For trained metrics, run a full training run and the benchmark scripts.")


if __name__ == "__main__":
    main()

