"""Quality evaluation: validation loss and perplexity for Baseline GPT vs H(AI)LP.

Evaluates on the same Simple English Wikipedia validation stream used in training.
Can run on randomly initialised models (baseline) or load from checkpoints.

Run:
    # Random init (no checkpoint) — baseline quality
    python benchmarks/quality_eval.py --model both --max-batches 50

    # After training — load best checkpoint per model
    python benchmarks/quality_eval.py --model both --checkpoint-dir checkpoints --max-batches 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.baseline_gpt import BaselineConfig, BaselineGPT
from models.hailp_model import HAILPConfig, HAILPModel
from training.data import get_dataloaders
from training.trainer import evaluate

# Match train.py model config (vocab_size for GPT-2 tokeniser)
VOCAB_SIZE = 50_257
SEQ_LEN = 256
BATCH_SIZE = 16


def _baseline_config() -> BaselineConfig:
    return BaselineConfig(
        vocab_size=VOCAB_SIZE,
        layers=6,
        attention_heads=8,
        hidden_dim=512,
        ffn_expansion=4,
        context_window=512,
        dropout=0.0,
    )


def _hailp_config() -> HAILPConfig:
    return HAILPConfig(
        vocab_size=VOCAB_SIZE,
        layers=12,
        hidden_dim=512,
        ffn_sharing_group_size=4,
        low_rank_dim=64,
        adapter_rank=32,
        dropout=0.0,
    )


def load_baseline(device: torch.device, checkpoint_path: Path | None = None) -> BaselineGPT:
    """Build Baseline GPT, optionally loading from checkpoint."""
    cfg = _baseline_config()
    model = BaselineGPT(cfg).to(device)
    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    return model


def load_hailp(device: torch.device, checkpoint_path: Path | None = None) -> HAILPModel:
    """Build H(AI)LP, optionally loading from checkpoint."""
    cfg = _hailp_config()
    model = HAILPModel(cfg).to(device)
    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    return model


def run_quality_eval(
    model_name: str,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    checkpoint_dir: Path | None,
    max_batches: int | None,
) -> dict[str, float]:
    """Evaluate one model; return loss and perplexity."""
    if model_name == "baseline":
        ckpt = (checkpoint_dir / "baseline" / "best.pt") if checkpoint_dir else None
        model = load_baseline(device, ckpt)
        is_recurrent = False
    else:
        ckpt = (checkpoint_dir / "hailp" / "best.pt") if checkpoint_dir else None
        model = load_hailp(device, ckpt)
        is_recurrent = True

    return evaluate(
        model,
        val_loader,
        device,
        is_recurrent=is_recurrent,
        max_batches=max_batches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H(AI)LP quality eval: validation loss & perplexity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "hailp", "both"),
        default="both",
        help="Which model(s) to evaluate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Root checkpoint dir (e.g. checkpoints); uses best.pt per model",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Max validation batches per model (None = all)",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=200,
        help="Total val batches to materialise (get_dataloaders)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="Device",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    print("Building validation dataloader (may download tokeniser / stream data)...")
    _, val_loader = get_dataloaders(
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        val_batches=args.val_batches,
    )
    print(f"  Device: {device}  |  max_batches: {args.max_batches}")
    print()

    models_to_eval = ["baseline", "hailp"] if args.model == "both" else [args.model]
    results: list[tuple[str, dict[str, float]]] = []

    for name in models_to_eval:
        print(f"Evaluating {name}...", end=" ")
        metrics = run_quality_eval(
            name,
            device,
            val_loader,
            args.checkpoint_dir,
            args.max_batches,
        )
        results.append((name, metrics))
        print(f"loss={metrics['loss']:.4f}  ppl={metrics['perplexity']:.2f}")

    print()
    print("=" * 50)
    print("Quality evaluation summary")
    print("=" * 50)
    header = f"  {'model':<12} | {'loss':>8} | {'perplexity':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in results:
        print(f"  {name:<12} | {m['loss']:>8.4f} | {m['perplexity']:>10.2f}")
    print()
    if args.checkpoint_dir is None:
        print("  (No checkpoint loaded — random init. Train then re-run with --checkpoint-dir.)")
    else:
        print(f"  Checkpoints loaded from: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
