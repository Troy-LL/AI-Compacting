# Kaggle Testing Setup

This document contains the specific code blocks and snippets we will use for testing our LLM Compact project in a Kaggle notebook environment.

## 1. Environment Check

Run this block to verify the GPU and PyTorch version available.

```python
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
```

## 2. Clone Repository

Clone the specific branch of the project repository into the Kaggle working directory.

```python
import subprocess
import shutil
import os

# Ensure we aren't currently "inside" the folder we are about to delete
os.chdir("/kaggle/working")

repo_path = "/kaggle/working/hailp"
if os.path.exists(repo_path):
    shutil.rmtree(repo_path)

result = subprocess.run([
    "git", "clone",
    "--branch", "Block2",
    "https://github.com/Troy-LL/AI-Compacting.git",
    repo_path
], capture_output=True, text=True)

print(result.stdout)
print(result.stderr)
```

## 3. Setup Path and Verify Files

Add the repository to the Python path and verify that key files exist.

```python
import sys
import os

# Add to path
sys.path.insert(0, "/kaggle/working/hailp")

# Verify key files exist
key_files = [
    "models/hailp_model.py",
    "training/trainer.py",
    "training/data.py",
    "training/device.py",
    "train.py",
]

for f in key_files:
    path = f"/kaggle/working/hailp/{f}"
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"{status} {f}")
```

## 4. Install Dependencies

Install the required Python packages for training.

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "datasets>=2.14.0",
    "tokenizers>=0.15.0", 
    "transformers>=4.35.0",
    "wandb>=0.16.0",
    "psutil>=5.9.0",
    "tqdm>=4.66.0",
    "accelerate>=0.24.0",
])
print("Dependencies installed")
```

## 5. Weights & Biases Login

Authenticate with your wandb account using your API key. By setting this purely via the environment, we avoid Jupyter kernel circular import issues!

```python
import os

# Set the key directly in the environment. 
# Your train.py script will automatically detect and use this when it launches later!
os.environ["WANDB_API_KEY"] = "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"
```

## 6. Configuration Override and Directory Setup

Change to the working directory, load the configuration, and override values for a quick sanity check.

```python
import os
import yaml

os.chdir("/kaggle/working/hailp")

# Override config for sanity check
with open("configs/hailp.yaml", "r") as f:
    config = yaml.safe_load(f)

# Temporarily override for quick test
config["total_steps"] = 250
config["batch_size"] = 32   # Per-GPU micro-batch (gradient accum compensates)
config["seq_len"] = 256
config["device"] = "cuda"
config["checkpoint_dir"] = "/kaggle/working/checkpoints"
config["checkpoint_every"] = 250
config["log_every"] = 10   # Already set to 10

os.makedirs(config["checkpoint_dir"], exist_ok=True)

print("Config:")
for k, v in config.items():
    print(f"  {k}: {v}")
```

## 7. Generate Multi-GPU Training Script

Run this cell to immediately generate the completely custom `train_multi.py` directly inside your Kaggle workspace without needing to use `git push`!

```python
%%writefile train_multi.py
#!/usr/bin/env python
"""train_multi.py -- Multi-GPU training with HuggingFace Accelerate.

T4-safe version that solves the Accelerate fp32-reconversion OOM.

Problem: Accelerate's DDP wrapper unconditionally converts ALL model
forward() outputs to fp32 AFTER the call.  With a 50k-vocab model at
batch=128, the logits tensor (128, 256, 50257) needs 6.14 GB in fp32.
Neither manual autocast nor `del logits` can prevent this because
Accelerate intercepts the output *before* our code runs.

Solution: Wrap the model in a LossWrapper that computes the cross-entropy
loss *inside* forward().  The wrapper returns only:
  - loss  (scalar — 4 bytes in fp32)
  - h_states  (list of (B, 512) tensors — ~6 MB total)
Accelerate's fp32 conversion now costs bytes, not gigabytes.
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
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


# ── LossWrapper ─────────────────────────────────────────────────────────
# Computes loss inside forward() so Accelerate never sees the giant logits.
class LossWrapper(nn.Module):
    """Thin wrapper that returns (loss, h_states) instead of (logits, h_states).

    This prevents Accelerate's automatic fp32 reconversion from
    duplicating the (B, T, 50257) logits tensor in fp32 (~6 GB at B=128).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, h_states=None):
        logits, h_states = self.model(x, h_states=h_states)
        loss = compute_loss(logits, y)
        # logits are never returned — they die here, saving ~6 GB
        return loss, h_states


# ── Config ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "sequence_length": 256,
    "batch_size": 32,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "warmup_steps": 500,
    "total_steps": 30_000,
    "mixed_precision": "fp16",  # T4 = Turing arch → fp16 only
    "log_every": 10,
    "checkpoint_every": 1_000,
    "val_batches": 200,
}

CHECKPOINT_ROOT = Path("checkpoints")


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
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
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.checkpoint_every is not None:
        config["checkpoint_every"] = args.checkpoint_every

    use_wandb = not args.no_wandb
    grad_accum = config["gradient_accumulation_steps"]

    # ----- ACCELERATE -----
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=grad_accum,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print("\nTraining configuration:")
        for k, v in config.items():
            print(f"  {k:30s}: {v}")
        eff_batch = config["batch_size"] * grad_accum * accelerator.num_processes
        print(f"  {'effective_global_batch':30s}: {eff_batch}")
        print(f"\n🚀 Using {accelerator.num_processes} GPUs/processes via Accelerate! 🚀\n")

    # ----- BUILD MODEL & DATA -----
    cfg = HAILPConfig(
        vocab_size=50_257, layers=12, hidden_dim=512,
        ffn_sharing_group_size=4, low_rank_dim=64, adapter_rank=32, dropout=0.1,
    )
    raw_model = HAILPModel(cfg)
    wrapped_model = LossWrapper(raw_model)

    train_loader, val_loader = get_dataloaders(
        seq_len=config["sequence_length"],
        batch_size=config["batch_size"],
        val_batches=config["val_batches"],
        device=device,
        num_workers=0,
    )

    optimizer = build_optimizer(raw_model, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=config["warmup_steps"], total_steps=config["total_steps"])

    # ----- PREPARE WITH ACCELERATE -----
    # We prepare the LossWrapper (not raw_model).  Accelerate wraps IT in DDP,
    # so the fp32 conversion only sees (loss_scalar, h_states) — not logits.
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        wrapped_model, optimizer, train_loader, val_loader
    )

    start_step = 0
    ckpt_mgr = CheckpointManager(CHECKPOINT_ROOT / "hailp_multi", keep_last=3)

    if args.resume:
        latest = ckpt_mgr.latest()
        if latest is not None:
            # Load into the inner HAILPModel (unwrap LossWrapper → DDP → LossWrapper → HAILPModel)
            inner = accelerator.unwrap_model(model).model
            start_step = ckpt_mgr.load(latest, inner, optimizer=optimizer, scaler=None, device=device)

    # ----- WANDB -----
    if use_wandb and _HAS_WANDB and accelerator.is_main_process:
        wandb.init(project="hailp-arch-comparison", name="hailp_multi",
                   resume="allow" if args.resume else None, config=config)

    # ----- TRAINING LOOP -----
    h_states = None
    step = start_step
    loss_accum = 0.0
    t0 = time.perf_counter()
    data_iter = iter(train_loader)

    while step < config["total_steps"]:
        model.train()
        scheduler.step(step)

        for micro in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            with accelerator.accumulate(model):
                # Forward through LossWrapper: returns (loss, h_states)
                # Logits are computed and discarded INSIDE the wrapper,
                # so Accelerate's fp32 conversion never touches them.
                loss, h_states = model(x, y, h_states=h_states)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config["gradient_clip"])

                optimizer.step()
                optimizer.zero_grad()

            if h_states is not None:
                h_states = [h.detach() for h in h_states]

            loss_accum += loss.detach().item()

        step += 1

        # Logging
        if step % config["log_every"] == 0 and accelerator.is_main_process:
            elapsed = time.perf_counter() - t0
            avg_loss = loss_accum / (config["log_every"] * grad_accum)
            ppl = math.exp(min(avg_loss, 20.0))
            eff_batch = config["batch_size"] * grad_accum * accelerator.num_processes
            tps = config["log_every"] * eff_batch * config["sequence_length"] / elapsed
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
            inner_model = accelerator.unwrap_model(model).model
            val_metrics = evaluate(inner_model, val_loader, device, is_recurrent=True)
            val_loss = val_metrics["loss"]

            print(f"\n  [val] step={step:,}  loss={val_loss:.4f}  ppl={val_metrics['perplexity']:.1f}\n")

            ckpt_mgr.save(step, inner_model, optimizer, val_loss, config, scaler=None)

            if use_wandb and _HAS_WANDB:
                wandb.log({"hailp/val_loss": val_loss, "hailp/val_perplexity": val_metrics["perplexity"]}, step=step)

    if use_wandb and _HAS_WANDB and getattr(accelerator, "is_main_process", True):
        wandb.finish()

if __name__ == "__main__":
    main()
```

## 8. Run Training (Multi-GPU)

Launch the training script using HuggingFace Accelerate so that it correctly spans across both your T4 GPUs!

```python
import sys
import os

os.chdir("/kaggle/working/hailp")

# batch-size 64 per GPU = effective global batch of 512 (64 × 4 accum × 2 GPUs)
# batch 128 OOMs because logits (128,256,50257) + cross_entropy softmax > 15GB T4 VRAM

!accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_multi.py \
    --steps 1000 \
    --batch-size 64 \
    --learning-rate 0.0006 \
    --checkpoint-every 50 \
    --resume

```
