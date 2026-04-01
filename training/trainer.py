"""Shared training utilities for Baseline GPT and H(AI)LP RWKV.

Public API (used by train.py and tests)
---------------------------------------
compute_loss          -- cross-entropy for next-token prediction
build_optimizer       -- AdamW with weight-decay param groups
WarmupCosineScheduler -- linear warmup then cosine decay
CheckpointManager     -- rotating saves + best-val tracking
ram_mb                -- current process RSS in MB
ram_at_seq            -- peak RAM delta for a forward pass at seq_len
train_step            -- single forward/backward/optimiser step
train_loop            -- full step-based loop (W&B, checkpointing, eval)

Legacy API (kept for test compatibility)
-----------------------------------------
train_one_epoch       -- epoch-based loop used by existing tests
evaluate              -- evaluation loop used by existing tests
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction.

    Parameters
    ----------
    logits  : Tensor of shape (B, T, vocab_size)
    targets : Tensor of shape (B, T) -- use -100 for padding positions
    """
    b, t, v = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(b * t, v),
        targets.reshape(b * t),
        ignore_index=ignore_index,
    )


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """AdamW with weight-decay on 2-D parameters only (biases/norms excluded).

    Follows the GPT-2 / nanoGPT convention: no decay on biases, LayerNorm
    weights, or any 1-D tensor.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay,   "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=betas)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay to ``min_lr_ratio * peak_lr``.

    Call ``scheduler.step(global_step)`` once per optimiser step (0-indexed).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]

    def get_multiplier(self, step: int) -> float:
        """Return the LR multiplier for ``step``."""
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self, step: int) -> None:
        """Apply the LR for the given global step to all param groups."""
        mult = self.get_multiplier(step)
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs, strict=True):
            group["lr"] = base_lr * mult


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Save / restore training state with rotation and best-val tracking.

    Keeps the ``keep_last`` most-recent checkpoints plus a separate
    ``best.pt`` for the lowest validation loss seen so far.

    Checkpoint dict schema::

        {
            "step":       int,
            "model":      state_dict,
            "optimizer":  state_dict,
            "scaler":     state_dict | None,
            "val_loss":   float,
            "config":     dict,
        }
    """

    def __init__(self, run_dir: str | Path, keep_last: int = 3) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._saved: list[Path] = []
        self._best_val = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        val_loss: float,
        config: dict,
        scaler: GradScaler | None = None,
    ) -> None:
        """Save a checkpoint and rotate old ones."""
        ckpt = {
            "step":      step,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict() if scaler is not None else None,
            "val_loss":  val_loss,
            "config":    config,
        }
        path = self.run_dir / f"ckpt_{step:07d}.pt"
        torch.save(ckpt, path)
        self._saved.append(path)

        # Rotate -- delete the oldest if we exceed keep_last
        if len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()

        # Track best
        if val_loss < self._best_val:
            self._best_val = val_loss
            torch.save(ckpt, self.run_dir / "best.pt")

        print(
            f"[ckpt] saved step={step:,}  "
            f"val_loss={val_loss:.4f}  best={self._best_val:.4f}  "
            f"path={path.name}"
        )

    def latest(self) -> Path | None:
        """Return the path of the newest checkpoint file, or None."""
        candidates = sorted(self.run_dir.glob("ckpt_*.pt"))
        return candidates[-1] if candidates else None

    def load(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scaler: GradScaler | None = None,
        device: torch.device | None = None,
    ) -> int:
        """Load a checkpoint; returns the saved step number."""
        if device is None:
            device = torch.device("cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        self._best_val = ckpt.get("val_loss", float("inf"))
        step: int = ckpt.get("step", 0)
        print(f"[ckpt] resumed step={step:,}  val_loss={self._best_val:.4f}")
        return step


# ---------------------------------------------------------------------------
# RAM measurement
# ---------------------------------------------------------------------------


def ram_mb() -> float:
    """Current process RSS in MB (CPU RAM).

    Uses psutil if available; falls back to CUDA allocated memory; 0.0 on
    bare CPU without psutil.
    """
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


@torch.no_grad()
def ram_at_seq(
    model: nn.Module,
    seq_len: int,
    batch_size: int = 1,
    device: torch.device | None = None,
    is_recurrent: bool = False,
    vocab_size: int = 50_257,
) -> float:
    """Peak RSS delta (MB) during a single forward pass at ``seq_len``.

    Snapshots RSS before and after the forward pass.  Simple but sufficient
    for comparing relative memory consumption between architectures.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    before = ram_mb()
    if is_recurrent:
        _ = model(x)
    else:
        _ = model(x)
    after = ram_mb()
    return max(after - before, 0.0)


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip: float = 1.0,
    scaler: GradScaler | None = None,
    is_recurrent: bool = False,
    h_states: list[torch.Tensor] | None = None,
) -> tuple[float, list[torch.Tensor] | None]:
    """One forward + backward + optimiser step.

    Returns
    -------
    loss_val : float
        Scalar loss value.
    h_states :
        Updated recurrent states (detached).  None for transformer models.
    """
    optimizer.zero_grad()

    if scaler is not None:
        with autocast():
            if is_recurrent:
                logits, h_states = model(x, h_states=h_states)
            else:
                logits = model(x)
            loss = compute_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        if is_recurrent:
            logits, h_states = model(x, h_states=h_states)
        else:
            logits = model(x)
        loss = compute_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    if is_recurrent and h_states is not None:
        h_states = [h.detach() for h in h_states]

    return loss.item(), h_states


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_recurrent: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate on a dataloader; return loss and perplexity.

    Parameters
    ----------
    max_batches :
        Stop early after this many batches.  None = evaluate everything.
    """
    model.eval()
    total_loss: float = 0.0
    steps: int = 0
    h_states: list[torch.Tensor] | None = None

    for i, (x, y) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if is_recurrent:
            logits, h_states = model(x, h_states=h_states)
            h_states = [h.detach() for h in h_states]
        else:
            logits = model(x)
        total_loss += compute_loss(logits, y).item()
        steps += 1

    avg_loss = total_loss / max(steps, 1)
    return {
        "loss":       avg_loss,
        "perplexity": float(math.exp(min(avg_loss, 20.0))),
    }


# ---------------------------------------------------------------------------
# Full step-based training loop
# ---------------------------------------------------------------------------


def train_loop(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
    checkpoint_dir: str | Path,
    is_recurrent: bool = False,
    start_step: int = 0,
    use_wandb: bool = False,
    run_id: str | None = None,
) -> None:
    """Step-based training loop with checkpointing, LR scheduling, and W&B.

    Parameters
    ----------
    model :
        Model to train (BaselineGPT or H(AI)LP).
    model_name :
        ``"baseline"`` or ``"hailp"`` -- used for checkpoint dir and W&B run.
    train_loader :
        Infinite streaming DataLoader.
    val_loader :
        Finite deterministic DataLoader for validation.
    device :
        Torch device.
    config :
        Training hyperparameters dict (see DEFAULT_CONFIG in train.py).
    checkpoint_dir :
        Root dir; checkpoints land in ``<checkpoint_dir>/<model_name>/``.
    is_recurrent :
        Pass True for H(AI)LP so h_states are threaded between steps.
    start_step :
        Resume from this step (0 for fresh runs).
    use_wandb :
        Log metrics to W&B if True and wandb is installed.
    run_id :
        W&B run ID for resuming a previous experiment.
    """
    model.to(device)

    optimizer = build_optimizer(
        model,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        total_steps=config["total_steps"],
    )

    # Mixed precision (GradScaler + CUDA autocast) is CUDA-only.
    use_amp = (
        config.get("mixed_precision") in ("bf16", "fp16")
        and device.type == "cuda"
    )
    scaler = GradScaler() if use_amp else None

    ckpt_mgr = CheckpointManager(
        run_dir=Path(checkpoint_dir) / model_name,
        keep_last=3,
    )

    # Weights & Biases
    if use_wandb and _HAS_WANDB:
        wandb.init(
            project="hailp-arch-comparison",
            name=model_name,
            id=run_id,
            resume="allow" if run_id else None,
            config=config,
        )

    # Training state
    h_states: list[torch.Tensor] | None = None
    step = start_step
    loss_accum = 0.0
    t0 = time.perf_counter()
    data_iter: Iterator = iter(train_loader)

    print(f"\n{'=' * 62}")
    print(f"  Training : {model_name}")
    print(f"  Device   : {device}")
    print(f"  Steps    : {start_step:,} -> {config['total_steps']:,}")
    print(f"  AMP      : {use_amp}   |   W&B : {use_wandb and _HAS_WANDB}")
    print(f"{'=' * 62}\n")

    while step < config["total_steps"]:
        model.train()
        scheduler.step(step)

        # --- fetch next batch (restart iterator on exhaustion) --------
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        loss_val, h_states = train_step(
            model, optimizer, x, y,
            grad_clip=config["gradient_clip"],
            scaler=scaler,
            is_recurrent=is_recurrent,
            h_states=h_states if is_recurrent else None,
        )

        loss_accum += loss_val
        step += 1

        # --- periodic console + W&B logging ---------------------------
        if step % config["log_every"] == 0:
            elapsed = time.perf_counter() - t0
            avg_loss = loss_accum / config["log_every"]
            ppl = math.exp(min(avg_loss, 20.0))
            seq = config["sequence_length"]
            tps = config["log_every"] * config["batch_size"] * seq / elapsed
            current_lr = optimizer.param_groups[0]["lr"]
            current_ram = ram_mb()

            print(
                f"step {step:6,}/{config['total_steps']:,} | "
                f"loss {avg_loss:.4f} | ppl {ppl:7.2f} | "
                f"lr {current_lr:.2e} | {tps:,.0f} tok/s | {current_ram:.0f} MB"
            )

            if use_wandb and _HAS_WANDB:
                wandb.log(
                    {
                        f"{model_name}/train_loss":        avg_loss,
                        f"{model_name}/train_perplexity":  ppl,
                        f"{model_name}/learning_rate":     current_lr,
                        f"{model_name}/tokens_per_second": tps,
                        f"{model_name}/ram_mb":            current_ram,
                    },
                    step=step,
                )

            loss_accum = 0.0
            t0 = time.perf_counter()

        # --- checkpoint + validation ----------------------------------
        if step % config["checkpoint_every"] == 0 or step == config["total_steps"]:
            val_metrics = evaluate(model, val_loader, device, is_recurrent=is_recurrent)
            val_loss = val_metrics["loss"]
            val_ppl  = val_metrics["perplexity"]

            # Memory profile at three representative sequence lengths
            _dev = device
            r64  = ram_at_seq(model, 64,  device=_dev, is_recurrent=is_recurrent)
            r256 = ram_at_seq(model, 256, device=_dev, is_recurrent=is_recurrent)
            r512 = ram_at_seq(model, 512, device=_dev, is_recurrent=is_recurrent)

            print(
                f"\n  [val] step={step:,}  loss={val_loss:.4f}  ppl={val_ppl:.1f}\n"
                f"  [ram] seq64={r64:.1f} MB  seq256={r256:.1f} MB  seq512={r512:.1f} MB\n"
            )

            ckpt_mgr.save(step, model, optimizer, val_loss, config, scaler)

            if use_wandb and _HAS_WANDB:
                wandb.log(
                    {
                        f"{model_name}/val_loss":        val_loss,
                        f"{model_name}/val_perplexity":  val_ppl,
                        f"{model_name}/ram_at_seq_64":   r64,
                        f"{model_name}/ram_at_seq_256":  r256,
                        f"{model_name}/ram_at_seq_512":  r512,
                    },
                    step=step,
                )

    if use_wandb and _HAS_WANDB:
        wandb.finish()

    print(f"\n  Training complete: {model_name} ({step:,} steps)")


# ---------------------------------------------------------------------------
# Legacy epoch-based API (kept for test compatibility)
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    is_recurrent: bool = False,
    grad_clip: float = 1.0,
    log_interval: int = 50,
) -> dict[str, float]:
    """Train for one full pass over the dataloader.

    Kept for test compatibility.  New code should use ``train_loop``.
    """
    model.train()
    total_loss = 0.0
    steps = 0
    h_states: list[torch.Tensor] | None = None
    t0 = time.perf_counter()

    for step, (input_ids, targets) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        loss_val, h_states = train_step(
            model, optimizer, input_ids, targets,
            grad_clip=grad_clip,
            is_recurrent=is_recurrent,
            h_states=h_states if is_recurrent else None,
        )
        total_loss += loss_val
        steps += 1

        if log_interval > 0 and (step + 1) % log_interval == 0:
            avg = total_loss / steps
            print(
                f"  step {step + 1:4d} | loss {avg:.4f} | "
                f"ppl {math.exp(min(avg, 20.0)):.2f}"
            )

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(steps, 1)
    return {
        "loss":          avg_loss,
        "perplexity":    float(math.exp(min(avg_loss, 20.0))),
        "steps_per_sec": steps / elapsed,
    }
