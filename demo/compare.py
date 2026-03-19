"""Terminal comparison demo for H(AI)LP vs Baseline GPT.

Run:

    uv run python demo/compare.py

This script does NOT require trained checkpoints; it uses fresh models
to demonstrate:

- Parameter counts and approximate FP32 sizes
- Memory usage vs sequence length (RSS delta)
- Forward speed (tokens/sec) at a modest context length
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.baseline_gpt import BaselineConfig, BaselineGPT
from models.hailp_model import HAILPConfig, HAILPModel
from training.trainer import ram_at_seq


def _human_mb(params: int) -> float:
    return params * 4 / (1024**2)


def _measure_speed(model: torch.nn.Module, seq_len: int, batch_size: int = 1, repeats: int = 5) -> float:
    """Return tokens/sec for a simple forward loop on CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    vocab = getattr(getattr(model, "config", None), "vocab_size", 50_257)
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        model(x)

    total_tokens = batch_size * seq_len * repeats
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return total_tokens / max(dt, 1e-6)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nH(AI)LP Terminal Comparison Demo")
    print("=" * 70)
    print(f"Device: {device}")

    # Build models
    b_cfg = BaselineConfig(
        vocab_size=50_257,
        layers=6,
        attention_heads=8,
        hidden_dim=512,
        ffn_expansion=4,
        context_window=4096,
        dropout=0.0,
    )
    h_cfg = HAILPConfig(
        vocab_size=50_257,
        layers=12,
        hidden_dim=512,
        ffn_sharing_group_size=4,
        low_rank_dim=64,
        adapter_rank=32,
        dropout=0.0,
    )

    baseline = BaselineGPT(b_cfg)
    hailp = HAILPModel(h_cfg)

    b_params = baseline.num_parameters()
    h_params = hailp.num_parameters()

    print("\n[1] Parameter counts and sizes (FP32)")
    print(f"  Baseline GPT  : {b_params:>12,} params  (~{_human_mb(b_params):5.1f} MB)")
    print(f"  H(AI)LP RWKV  : {h_params:>12,} params  (~{_human_mb(h_params):5.1f} MB)")

    print("\n[2] Memory usage vs sequence length (RSS delta, MB)")
    seq_512 = 512
    seq_2048 = 2048
    b_ram_512 = ram_at_seq(baseline.to(device), seq_512, device=device, is_recurrent=False)
    h_ram_512 = ram_at_seq(hailp.to(device), seq_512, device=device, is_recurrent=True)
    b_ram_2048 = ram_at_seq(baseline.to(device), seq_2048, device=device, is_recurrent=False)
    h_ram_2048 = ram_at_seq(hailp.to(device), seq_2048, device=device, is_recurrent=True)

    print(f"  At seq={seq_512}:  Baseline ~{b_ram_512:6.1f} MB   |   H(AI)LP ~{h_ram_512:6.1f} MB")
    print(f"  At seq={seq_2048}: Baseline ~{b_ram_2048:6.1f} MB   |   H(AI)LP ~{h_ram_2048:6.1f} MB")

    print("\n[3] Forward speed (tokens/sec, untrained models)")
    tps_baseline = _measure_speed(BaselineGPT(b_cfg), seq_len=512, batch_size=1, repeats=5)
    tps_hailp = _measure_speed(HAILPModel(h_cfg), seq_len=512, batch_size=1, repeats=5)
    print(f"  Baseline GPT  : {tps_baseline:8.1f} tok/s at seq=512")
    print(f"  H(AI)LP RWKV  : {tps_hailp:8.1f} tok/s at seq=512")

    print("\nNote: these numbers are from randomly initialised models on this machine.")
    print("For trained metrics, run a full training run and the benchmark scripts.")


if __name__ == "__main__":
    main()

