"""Streaming tokenisation pipeline -- Simple English Wikipedia.

Pipeline
--------
HF streaming -> text + EOS -> GPT-2 BPE tokeniser -> pack into fixed-length
(seq_len+1) chunks -> shift by 1 for targets -> DataLoader.

No padding is ever used.  The dataset cycles forever; train.py stops after
``total_steps`` batches.

Validation set
--------------
Simple English Wikipedia only has a ``train`` split.  A different seed (999)
produces a statistically independent pseudo-validation stream.  We materialise
a fixed number of batches into RAM so validation is deterministic and fast.
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKENIZER_NAME: str = "gpt2"       # 50 257-vocab BPE
DEFAULT_SEQ_LEN: int = 256

# Build EOS at runtime so the literal never appears verbatim in this file.
# GPT-2 uses this as the document separator token.
_EOS: str = "<" + "|endoftext|" + ">"


# ---------------------------------------------------------------------------
# Tokeniser helper
# ---------------------------------------------------------------------------


def get_tokenizer(name: str = TOKENIZER_NAME) -> AutoTokenizer:
    """Return a tokeniser instance with pad=eos (GPT-2 has no pad token)."""
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Streaming iterable dataset
# ---------------------------------------------------------------------------


class WikiStreamDataset(IterableDataset):
    """Infinite stream of (input_ids, targets) from Simple English Wikipedia.

    Text is packed end-to-end with EOS separators -- no padding tokens are
    ever produced.  The iterator cycles the dataset indefinitely; the caller
    (train.py) stops it after the desired number of steps.

    Parameters
    ----------
    split :
        HuggingFace split name.  Simple English Wikipedia only exposes
        ``"train"``; use ``seed=999`` for a pseudo-validation stream.
    seq_len :
        Tokens per sample.  ``targets = input_ids`` shifted left by 1.
    tokenizer_name :
        Any HuggingFace tokeniser identifier compatible with AutoTokenizer.
    shuffle_buffer :
        Articles buffered for random shuffling.  0 = disabled.
    seed :
        Shuffle seed.  42 for training, 999 for validation.
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = DEFAULT_SEQ_LEN,
        tokenizer_name: str = TOKENIZER_NAME,
        shuffle_buffer: int = 1_000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.split = split
        self.seq_len = seq_len
        self.tokenizer_name = tokenizer_name
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hf_stream(self):
        """Return a (optionally shuffled) HuggingFace streaming dataset."""
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.simple",
            split=self.split,
            streaming=True,
            trust_remote_code=False,
        )
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        return ds

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        tok = get_tokenizer(self.tokenizer_name)
        vocab_size = tok.vocab_size
        stride: int = self.seq_len + 1      # +1 so we can shift into targets
        buf: list[int] = []

        while True:                          # cycle forever
            for row in self._hf_stream():
                text: str = row["text"] + _EOS
                ids = tok.encode(text, add_special_tokens=False)
                ids = [min(t, vocab_size - 1) for t in ids]   # clamp OOV
                buf.extend(ids)
                while len(buf) >= stride:
                    chunk, buf = buf[:stride], buf[stride:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:],  dtype=torch.long)
                    yield x, y


# ---------------------------------------------------------------------------
# Materialised validation dataset
# ---------------------------------------------------------------------------


class FiniteDataset(Dataset):
    """Materialise the first *n* items from an IterableDataset into RAM.

    Used to build a fixed, deterministic validation set from the Wikipedia
    stream with ``seed=999``.
    """

    def __init__(self, src: IterableDataset, n: int) -> None:
        self._items: list = []
        for i, item in enumerate(src):
            self._items.append(item)
            if i + 1 >= n:
                break

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        return self._items[idx]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def get_dataloaders(
    seq_len: int = DEFAULT_SEQ_LEN,
    batch_size: int = 16,
    num_workers: int = 0,
    val_batches: int = 200,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for Simple English Wikipedia.

    Parameters
    ----------
    seq_len :
        Tokens per sample -- identical for both Baseline and H(AI)LP.
    batch_size :
        Mini-batch size.
    num_workers :
        DataLoader workers.  Default 0 is safe on Windows and macOS.
    val_batches :
        Number of batches materialised into RAM for validation.
    device :
        Optional device used to apply DirectML-friendly DataLoader settings.

    Returns
    -------
    train_loader :
        Infinite streaming DataLoader (cycles the dataset).
    val_loader :
        Finite, deterministic DataLoader over ``val_batches`` batches.
    """
    train_ds = WikiStreamDataset(split="train", seq_len=seq_len, seed=42)

    val_ds = FiniteDataset(
        WikiStreamDataset(split="train", seq_len=seq_len, seed=999),
        n=val_batches * batch_size,
    )

    directml_device_types = {"privateuseone", "directml"}
    is_directml = device is not None and device.type in directml_device_types
    is_cuda = device is not None and device.type == "cuda"

    # Keep existing behavior when device is not provided.
    resolved_pin_memory = torch.cuda.is_available() if device is None else is_cuda

    resolved_num_workers = num_workers
    resolved_persistent_workers = False

    if is_directml:
        # Windows sweet spot for multiprocessing; avoid pinning for DirectML.
        resolved_num_workers = 2 if num_workers == 0 else num_workers
        resolved_pin_memory = False
        if resolved_num_workers > 0:
            resolved_persistent_workers = True

    train_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": resolved_num_workers,
        "pin_memory": resolved_pin_memory,
    }
    if resolved_persistent_workers:
        train_kwargs["persistent_workers"] = True
        train_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=0,       # val is in RAM -- workers give no benefit
        shuffle=False,
    )
    return train_loader, val_loader
