"""Cross-layer FFN weight sharing module.

Instead of each transformer layer having its own FFN, we maintain a pool of
SharedFFN blocks and assign layers to share them. This reduces parameters by a
factor equal to the group size.

Example with 12 layers and group_size=4:
    Layer  0: uses SharedFFN block 0
    Layer  4: uses SharedFFN block 0  ← same weights, not a copy!
    Layer  8: uses SharedFFN block 0
    Layer  1: uses SharedFFN block 1
    Layer  5: uses SharedFFN block 1  ← same weights, not a copy!
    ...
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedFFN(nn.Module):
    """A feed-forward block whose weights are shared across multiple layers.

    Architecture identical to a standard transformer FFN:
        FFN(x) = W2 * GELU(W1 * x + b1) + b2

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension (input and output).
    ffn_dim : int
        Intermediate projection dimension (typically 4 * hidden_dim).
    dropout : float
        Dropout probability applied after the activation.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN: x → up-project → GELU → dropout → down-project."""
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, ffn_dim={self.ffn_dim}"


class SharedFFNPool(nn.Module):
    """Pool of SharedFFN blocks assigned to layers by group.

    Assignment rule:
        block_index = layer_index % group_size
        (layers 0, group_size, 2*group_size, ... share block 0)
        (layers 1, group_size+1, 2*group_size+1, ... share block 1)

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension.
    ffn_dim : int
        FFN intermediate dimension.
    num_layers : int
        Total number of transformer layers.
    group_size : int
        Number of layers per shared FFN block.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_layers: int,
        group_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if group_size <= 0 or group_size > num_layers:
            raise ValueError(
                f"group_size must be in [1, num_layers], got {group_size}"
            )

        self.group_size = group_size
        self.num_blocks = group_size  # one block per position within a group

        # Only `group_size` unique FFN blocks regardless of num_layers
        self.blocks = nn.ModuleList([
            SharedFFN(hidden_dim, ffn_dim, dropout)
            for _ in range(self.num_blocks)
        ])

    def get_ffn_for_layer(self, layer_idx: int) -> SharedFFN:
        """Return the SharedFFN block assigned to `layer_idx`.

        Layers 0, group_size, 2*group_size use block 0.
        Layers 1, group_size+1, 2*group_size+1 use block 1.
        Etc.
        """
        return self.blocks[layer_idx % self.group_size]
