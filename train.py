#!/usr/bin/env python
"""train.py -- H(AI)LP training loop (no baseline).

H(AI)LP (RWKV-style recurrent state) is trained on the same streaming
Simple English Wikipedia token stream used by the original project.

Usage
-----
# Train H(AI)LP from scratch:
    python train.py

# Resume from the latest checkpoint:
    python train.py --resume

# Disable Weights & Biases (offline / no account):
    python train.py --no-wandb

# Quick smoke test -- 200 steps, batch size 4:
    python train.py --steps 200 --batch-size 4 --no-wandb

Checkpoints are saved to::

    checkpoints/<model_name>/ckpt_XXXXXXX.pt   (rotating, keep last 3)
    checkpoints/<model_name>/best.pt            (lowest val loss)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler

from models.hailp_model import HAILPConfig, HAILPModel
from training.device import DEVICE as DEFAULT_DEVICE
from training.data import get_dataloaders
from training.trainer import (
    CheckpointManager,
    build_optimizer,
    train_loop,
)

# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    # Data
    "sequence_length":   256,
    "batch_size":        16,
    # Optimiser
    "learning_rate":     3e-4,
    "weight_decay":      0.1,
    "gradient_clip":     1.0,
    "optimizer":         "AdamW",
    # LR schedule
    "warmup_steps":      500,
    "total_steps":       30_000,
    # Precision
    "mixed_precision":   "bf16",    # ignored if no CUDA
    # Logging & checkpointing
    "log_every":         100,
    "checkpoint_every":  1_000,
    "val_batches":       200,       # batches used for each validation eval
}

CHECKPOINT_ROOT = Path("checkpoints")


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def make_hailp() -> tuple[torch.nn.Module, bool]:
    """Build H(AI)LP RWKV model; return (model, is_recurrent=True)."""
    cfg = HAILPConfig(
        vocab_size=50_257,
        layers=12,
        hidden_dim=512,
        ffn_sharing_group_size=4,
        low_rank_dim=64,
        adapter_rank=32,
        dropout=0.1,
    )
    model = HAILPModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HAILPModel  : {n_params / 1e6:.1f}M parameters")
    return model, True


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: object,
    device: torch.device,
) -> int:
    """Load the latest checkpoint if one exists; return start_step."""
    model_name = "hailp"
    ckpt_dir = CHECKPOINT_ROOT / model_name
    mgr = CheckpointManager(ckpt_dir)
    latest = mgr.latest()
    if latest is None:
        print(f"[{model_name}] No checkpoint found -- starting from scratch.")
        return 0
    step = mgr.load(latest, model, optimizer, scaler, device)
    return step


# ---------------------------------------------------------------------------
# Single-model training entry point
# ---------------------------------------------------------------------------


def run_model(
    device: torch.device,
    config: dict,
    resume: bool,
    use_wandb: bool,
) -> None:
    """Build, (optionally resume), and train one model."""
    model_name = "hailp"
    print(f"\n{'#' * 62}")
    print(f"  Model : {model_name.upper()}")
    print(f"{'#' * 62}")

    # Build model
    model, is_recurrent = make_hailp()

    # Build dataloaders (identical for both models)
    print("Building dataloaders (materialising validation set)...")
    t0 = time.perf_counter()
    train_loader, val_loader = get_dataloaders(
        seq_len=config["sequence_length"],
        batch_size=config["batch_size"],
        val_batches=config["val_batches"],
        device=device,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Resume
    start_step = 0
    if resume:
        use_amp = (
            config.get("mixed_precision") in ("bf16", "fp16")
            and device.type == "cuda"
        )
        optimizer = build_optimizer(
            model,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scaler = GradScaler() if use_amp else None
        start_step = maybe_resume(model, optimizer, scaler, device)
        if start_step >= config["total_steps"]:
            print(
                f"[{model_name}] Already complete "
                f"({start_step:,} >= {config['total_steps']:,} steps). Skipping."
            )
            return

    # Train
    train_loop(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        checkpoint_dir=CHECKPOINT_ROOT,
        is_recurrent=is_recurrent,
        start_step=start_step,
        use_wandb=use_wandb,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train H(AI)LP on Simple English Wikipedia.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in checkpoints/<model>/.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total_steps (handy for smoke tests, e.g. --steps 200).",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size.",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override sequence_length.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = DEFAULT_DEVICE

    # Build config (start from defaults, apply overrides)
    config = DEFAULT_CONFIG.copy()
    if args.steps is not None:
        config["total_steps"] = args.steps
        config["warmup_steps"] = min(config["warmup_steps"], args.steps // 10)
        config["checkpoint_every"] = min(
            config["checkpoint_every"], max(args.steps // 5, 1)
        )
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seq_len is not None:
        config["sequence_length"] = args.seq_len

    use_wandb = not args.no_wandb

    print("\nTraining configuration:")
    for k, v in config.items():
        print(f"  {k:20s}: {v}")

    run_model(device, config, args.resume, use_wandb)

    print("\n  All done.")


if __name__ == "__main__":
    main()
