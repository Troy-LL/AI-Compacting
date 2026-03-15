"""Shared training loop utilities for both Baseline GPT and LifeLink RWKV.

Usage
-----
from training.trainer import train_one_epoch, evaluate

# For Baseline GPT:
train_one_epoch(model=baseline, optimizer=opt, dataloader=dl, device=device)

# For LifeLink RWKV:
train_one_epoch(model=lifelink, optimizer=opt, dataloader=dl, device=device,
                is_recurrent=True)
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction.

    Parameters
    ----------
    logits : Tensor, shape (B, T, vocab_size)
    targets : Tensor, shape (B, T) — next-token IDs, -100 for padding
    ignore_index : int
        Token ID to ignore in loss (default -100, PyTorch convention).

    Returns
    -------
    loss : scalar Tensor
    """
    B, T, V = logits.shape
    loss = nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=ignore_index,
    )
    return loss


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

    Parameters
    ----------
    model : nn.Module
        Either BaselineGPT or LifeLinkRWKV.
    optimizer : Optimizer
    dataloader : DataLoader
        Yields (input_ids, target_ids) tuples, both shape (B, T).
    device : torch.device
    is_recurrent : bool
        If True, passes h_states between batches within an epoch.
        Set True for LifeLink RWKV to demonstrate stateful inference.
    grad_clip : float
        Max gradient norm.
    log_interval : int
        Print loss every N steps.

    Returns
    -------
    dict with 'loss', 'perplexity', 'steps_per_sec'
    """
    model.train()
    total_loss = 0.0
    steps = 0
    h_states = None
    t0 = time.perf_counter()

    for step, (input_ids, targets) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if is_recurrent:
            logits, h_states = model(input_ids, h_states=h_states)
            # Detach states between batches to prevent backprop through entire history
            h_states = [h.detach() for h in h_states]
        else:
            logits = model(input_ids)

        loss = compute_loss(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if log_interval > 0 and (step + 1) % log_interval == 0:
            avg_loss = total_loss / steps
            print(f"  step {step+1:4d} | loss {avg_loss:.4f} | ppl {avg_loss:.2f}")

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(steps, 1)

    return {
        "loss": avg_loss,
        "perplexity": float(torch.exp(torch.tensor(avg_loss))),
        "steps_per_sec": steps / elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_recurrent: bool = False,
) -> dict[str, float]:
    """Evaluate model on a dataloader.

    Returns
    -------
    dict with 'loss' and 'perplexity'
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    h_states = None

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        if is_recurrent:
            logits, h_states = model(input_ids, h_states=h_states)
            h_states = [h.detach() for h in h_states]
        else:
            logits = model(input_ids)

        loss = compute_loss(logits, targets)
        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / max(steps, 1)
    return {
        "loss": avg_loss,
        "perplexity": float(torch.exp(torch.tensor(avg_loss))),
    }
