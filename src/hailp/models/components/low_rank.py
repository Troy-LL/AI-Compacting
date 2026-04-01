"""Low-rank linear projection module.

Decomposes a weight matrix W (out_features × in_features) into
two smaller matrices:

    y = x @ V^T @ U^T  ≡  y = x @ W^T   where W ≈ U @ V^T

Parameters
----------
in_features : int
    Input dimension.
out_features : int
    Output dimension.
rank : int
    Rank of the decomposition. Must be ≤ min(in_features, out_features).
    Reduces parameter count from (in_features × out_features) to
    rank × (in_features + out_features).
bias : bool
    Whether to add a learnable bias term.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Low-rank factorised linear layer: y = x @ V^T @ U^T + bias.

    Parameter count: rank * (in_features + out_features) + bias.
    Compare with full Linear: in_features * out_features + bias.

    Example
    -------
    >>> lrl = LowRankLinear(512, 512, rank=64)
    >>> x = torch.randn(2, 16, 512)
    >>> lrl(x).shape
    torch.Size([2, 16, 512])
    >>> lrl.effective_rank
    64
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank {rank} must be ≤ min(in={in_features}, out={out_features}) = "
                f"{min(in_features, out_features)}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.effective_rank = rank

        # W ≈ U @ V^T where U is (out_features × rank), V is (in_features × rank)
        # Forward:  y = (x @ V) @ U^T
        self.V = nn.Linear(in_features, rank, bias=False)   # down-project
        self.U = nn.Linear(rank, out_features, bias=bias)    # up-project

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation scaled to approximate full-rank variance."""
        nn.init.kaiming_uniform_(self.V.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U.weight, a=math.sqrt(5))
        if self.U.bias is not None:
            fan_in = self.effective_rank
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.U.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x → (x @ V) → U → output."""
        return self.U(self.V(x))

    def num_parameters(self) -> int:
        """Return total trainable parameters in this module."""
        return sum(p.numel() for p in self.parameters())

    def compression_ratio(self) -> float:
        """Return ratio of full-rank params to actual params (higher = more compressed)."""
        full_rank_params = self.in_features * self.out_features
        return full_rank_params / self.num_parameters()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.effective_rank}"
        )
