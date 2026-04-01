"""Language adapter module.

Small bottleneck MLP injected at each layer boundary. In a multi-lingual or
multi-domain setting, adapters allow fast task-specific fine-tuning by only
updating the adapter weights (~1-2% of parameters) while freezing the
backbone.

Architecture:
    x  →  LayerNorm  →  down-project(rank)  →  GELU  →  up-project(hidden)  →  + x

This is a residual adapter: it learns a delta on top of the frozen backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageAdapter(nn.Module):
    """Bottleneck adapter with residual connection.

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension (input and output).
    adapter_rank : int
        Bottleneck dimension. Typically 16–64. Smaller = fewer parameters
        but less expressive.
    dropout : float
        Dropout probability inside the adapter.

    Example
    -------
    >>> adapter = LanguageAdapter(hidden_dim=512, adapter_rank=32)
    >>> x = torch.randn(2, 16, 512)
    >>> adapter(x).shape
    torch.Size([2, 16, 512])
    """

    def __init__(
        self,
        hidden_dim: int,
        adapter_rank: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if adapter_rank <= 0:
            raise ValueError(f"adapter_rank must be positive, got {adapter_rank}")
        if adapter_rank > hidden_dim:
            raise ValueError(
                f"adapter_rank {adapter_rank} must be ≤ hidden_dim {hidden_dim}"
            )

        self.hidden_dim = hidden_dim
        self.adapter_rank = adapter_rank

        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, adapter_rank)
        self.up = nn.Linear(adapter_rank, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise down to near-zero so adapter starts as identity."""
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)   # up starts at zero → adapter is identity at init
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter with residual: x + adapter(norm(x))."""
        residual = x
        h = self.norm(x)
        h = self.down(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.up(h)
        return residual + h

    def num_parameters(self) -> int:
        """Return total trainable parameters in this adapter."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, adapter_rank={self.adapter_rank}"
