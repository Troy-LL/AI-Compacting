#!/usr/bin/env python
"""train_multi.py -- Multi-GPU supported training loop using HuggingFace Accelerate."""

import argparse
import math
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator

from models.hailp_model import HAILPConfig, HAILPModel
from training.data import get_dataloaders
from training.trainer import (
    CheckpointManager,
    WarmupCosineScheduler,
    build_optimizer,
    compute_loss,
    ram_mb,
    evaluate
)

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

# Copy DEFAULT_CONFIG from train.py
DEFAULT_CONFIG = {
    "sequence_length": 256,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "warmup_steps": 500,
    "total_steps": 30_000,
    "mixed_precision": "bf16",
    "log_every": 100,
    "checkpoint_every": 1_000,
    "val_batches": 200,
}

CHECKPOINT_ROOT = Path("checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.steps is not None:
        config["total_steps"] = args.steps
        config["warmup_steps"] = min(config["warmup_steps"], args.steps // 10)
        config["checkpoint_every"] = min(config["checkpoint_every"], max(args.steps // 5, 1))
        config["val_batches"] = min(config["val_batches"], max(10, args.steps // 10))
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seq_len is not None:
        config["sequence_length"] = args.seq_len

    use_wandb = not args.no_wandb

    # ----- ACCELERATE INITIALIZATION -----
    # Accelerate replaces DataParallel, GradScaler, and strict Device mapping!
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    if accelerator.is_main_process:
        print("\nTraining configuration:")
        for k, v in config.items():
            print(f"  {k:20s}: {v}")
        print(f"\n🚀 Using {accelerator.num_processes} GPUs/processes via Accelerate! 🚀\n")

    # ----- BUILD MODEL & DATA -----
    cfg = HAILPConfig(layers=12, hidden_dim=512, ffn_sharing_group_size=4, low_rank_dim=64, adapter_rank=32, dropout=0.1)
    model = HAILPModel(cfg)
    is_recurrent = True

    train_loader, val_loader = get_dataloaders(
        seq_len=config["sequence_length"],
        batch_size=config["batch_size"],
        val_batches=config["val_batches"],
        device=device,
        num_workers=2, # Hardcoded bump for Kaggle multiprocessing performance
    )

    optimizer = build_optimizer(model, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=config["warmup_steps"], total_steps=config["total_steps"])

    # ----- PREPARE WITH ACCELERATE -----
    # Accelerate safely wraps models and handles all gradient shard syncs behind the scenes.
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    start_step = 0
    ckpt_mgr = CheckpointManager(CHECKPOINT_ROOT / "hailp_multi", keep_last=3)
    
    if args.resume:
        latest = ckpt_mgr.latest()
        if latest is not None:
            # We must load into the unwrapped model initially
            unwrapped_model = accelerator.unwrap_model(model)
            start_step = ckpt_mgr.load(latest, unwrapped_model, optimizer=optimizer, scaler=None, device=device)

    # ----- WANDB -----
    if use_wandb and _HAS_WANDB and accelerator.is_main_process:
        wandb.init(project="hailp-arch-comparison", name="hailp_multi", resume="allow" if args.resume else None, config=config)

    # ----- TRAINING LOOP -----
    h_states = None
    step = start_step
    loss_accum = 0.0
    t0 = time.perf_counter()
    data_iter = iter(train_loader)

    while step < config["total_steps"]:
        model.train()
        scheduler.step(step)

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        optimizer.zero_grad()

        # Model forward
        if is_recurrent:
            logits, h_states = model(x, h_states=h_states)
        else:
            logits = model(x)

        loss = compute_loss(logits, y)

        # Accelerate handles mixed precision and gradient synchronization automatically!
        accelerator.backward(loss)
        
        # Clip norms (using accelerate safe method)
        if hasattr(accelerator, "sync_gradients") and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            
        optimizer.step()

        if is_recurrent and h_states is not None:
            h_states = [h.detach() for h in h_states]

        # Since loss is calculated locally per GPU process, gather it to get the true average
        gathered_loss = accelerator.gather(loss)
        loss_val = gathered_loss.mean().item()
        
        loss_accum += loss_val
        step += 1

        # Logging (Main process only)
        if step % config["log_every"] == 0 and accelerator.is_main_process:
            elapsed = time.perf_counter() - t0
            avg_loss = loss_accum / config["log_every"]
            ppl = math.exp(min(avg_loss, 20.0))
            tps = config["log_every"] * config["batch_size"] * config["sequence_length"] * accelerator.num_processes / elapsed
            current_lr = optimizer.param_groups[0]["lr"]
            current_ram = ram_mb()

            print(f"step {step:6,}/{config['total_steps']:,} | loss {avg_loss:.4f} | ppl {ppl:7.2f} | lr {current_lr:.2e} | {tps:,.0f} tok/s | {current_ram:.0f} MB")

            if use_wandb and _HAS_WANDB:
                wandb.log({
                    "hailp/train_loss": avg_loss,
                    "hailp/train_perplexity": ppl,
                    "hailp/learning_rate": current_lr,
                    "hailp/tokens_per_second": tps,
                }, step=step)

            loss_accum = 0.0
            t0 = time.perf_counter()

        # Checkpoint / Eval
        if (step % config["checkpoint_every"] == 0 or step == config["total_steps"]) and accelerator.is_main_process:
            # We explicitly unwrap the model to evaluate and save so that weights correspond perfectly to original names!
            unwrapped_model = accelerator.unwrap_model(model)
            val_metrics = evaluate(unwrapped_model, val_loader, device, is_recurrent=True)
            val_loss = val_metrics["loss"]
            
            print(f"\n  [val] step={step:,}  loss={val_loss:.4f}  ppl={val_metrics['perplexity']:.1f}\n")
            
            ckpt_mgr.save(step, unwrapped_model, optimizer, val_loss, config, scaler=None)
            
            if use_wandb and _HAS_WANDB:
                wandb.log({"hailp/val_loss": val_loss, "hailp/val_perplexity": val_metrics["perplexity"]}, step=step)

    if use_wandb and _HAS_WANDB and getattr(accelerator, "is_main_process", True):
        wandb.finish()

if __name__ == "__main__":
    main()
