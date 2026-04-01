"""H(AI)LP Model — recurrent architecture with fixed memory footprint.

The key innovation is replacing the growing KV cache with a fixed-size
recurrent hidden state. Regardless of sequence length, memory is constant.

Architecture differences vs Baseline GPT
-----------------------------------------
1. RWKV time-mixing instead of full self-attention:
   - O(n) per token instead of O(n²) per sequence
   - No KV cache — uses a fixed-size state tensor h_state

2. Parameter sharing:
   - FFN weights shared across groups of `ffn_sharing_group_size` layers
   - 12 layers but only 4 unique FFN blocks (group_size=4)

3. Low-rank attention projections:
   - W_Q, W_K, W_V decomposed as U @ V^T where rank=64
   - Reduces projection parameters by ~4× vs full rank

4. Language adapters:
   - Small bottleneck MLPs at each layer boundary
   - Enable fast domain adaptation without full fine-tuning

Memory behaviour
----------------
At seq=64:   h_state shape = (B, layers, hidden_dim)  [fixed]
At seq=512:  h_state shape = (B, layers, hidden_dim)  [IDENTICAL — fixed!]

This is the point. No matter how long the context, memory is constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .components.adapter import LanguageAdapter
from .components.low_rank import LowRankLinear
from .components.param_sharing import SharedFFNPool


@dataclass
class HAILPConfig:
    """Configuration for the H(AI)LP model."""

    layers: int = 12
    hidden_dim: int = 512
    vocab_size: int = 8000
    ffn_sharing_group_size: int = 4
    low_rank_dim: int = 64
    adapter_rank: int = 32
    dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: str | Path) -> HAILPConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            layers=cfg.get("layers", 12),
            hidden_dim=cfg.get("hidden_dim", 512),
            vocab_size=cfg.get("vocab_size", 8000),
            ffn_sharing_group_size=cfg.get("ffn_sharing_group_size", 4),
            low_rank_dim=cfg.get("low_rank_dim", 64),
            adapter_rank=cfg.get("adapter_rank", 32),
        )

    @property
    def state_bytes(self) -> int:
        """Fixed state size in bytes (float32) — independent of sequence length."""
        # h_state: (batch=1, layers, hidden_dim) × float32
        return 1 * self.layers * self.hidden_dim * 4


class RWKVTimeMixing(nn.Module):
    """RWKV-style time mixing — O(n) recurrent attention replacement.

    Each token is processed one at a time. The hidden state `h_state`
    accumulates context via an exponential moving average (EMA):

        h_t = decay * h_{t-1} + k_t * v_t    [wkv recurrence]
        y_t = softmax(q_t) * h_t

    This is RWKV's "WKV" operator, simplified for clarity. The key property:
    h_state is a FIXED-SIZE tensor (hidden_dim) regardless of how many tokens
    have been processed.

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension.
    rank : int
        Rank for low-rank Q/K/V projections.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # Low-rank projections for Q, K, V (the "time-mixing" projections)
        self.w_q = LowRankLinear(hidden_dim, hidden_dim, rank=rank)
        self.w_k = LowRankLinear(hidden_dim, hidden_dim, rank=rank)
        self.w_v = LowRankLinear(hidden_dim, hidden_dim, rank=rank)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable time decay (one value per channel)
        # Initialised to small negative values so decay starts near 1 (slow forgetting)
        self.time_decay = nn.Parameter(torch.ones(hidden_dim) * -0.5)

        # Time shift (RWKV lerp between current token and previous token).
        # NOTE: Avoid padding ops with mixed positive/negative values because
        # torch-directml has current limitations here.

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence with optional recurrent state.

        Parameters
        ----------
        x : Tensor, shape (B, T, hidden_dim)
            Input hidden states.
        h_state : Tensor or None, shape (B, hidden_dim)
            Recurrent state from previous call. If None, initialised to zeros.
            THIS IS THE FIXED-SIZE STATE — shape does NOT depend on T.

        Returns
        -------
        out : Tensor, shape (B, T, hidden_dim)
        h_state : Tensor, shape (B, hidden_dim)
            Updated recurrent state for the next call.
        """
        b, t, c = x.shape

        # Initialise state if needed
        if h_state is None:
            h_state = torch.zeros(b, c, device=x.device, dtype=x.dtype)

        x = self.ln(x)

        # Time-shift: mix current token with previous token embeddings.
        # Equivalent to shifting sequence dim by 1 and inserting zeros at t=0.
        x_shifted = torch.cat(
            [
                torch.zeros(b, 1, c, device=x.device, dtype=x.dtype),
                x[:, :-1, :],
            ],
            dim=1,
        )  # (b, t, c)
        mix = 0.5 * x + 0.5 * x_shifted          # simple lerp (RWKV uses learned mix)

        q = self.w_q(mix)                         # (b, t, c)
        k = self.w_k(mix)                         # (b, t, c)
        v = self.w_v(mix)                         # (b, t, c)

        # Decay coefficient: exp(-exp(time_decay)) ∈ (0, 1)
        decay = torch.exp(-torch.exp(self.time_decay))  # (c,)

        # Recurrent WKV over the sequence dimension
        outputs = []
        for _t in range(t):
            # Update state: exponential moving average
            h_state = decay * h_state + k[:, _t, :] * v[:, _t, :]  # (b, c)
            # Query the state
            y = F.sigmoid(q[:, _t, :]) * h_state                  # (b, c)
            outputs.append(y.unsqueeze(1))

        out = torch.cat(outputs, dim=1)           # (b, t, c)
        out = self.dropout(self.out_proj(out))

        return out, h_state.detach()


class HAILPBlock(nn.Module):
    """H(AI)LP transformer block.

    Differs from a standard GPT block in:
    1. Uses RWKVTimeMixing instead of full self-attention
    2. Uses a SharedFFN (same weights as other layers in its group)
    3. Has a LanguageAdapter after the FFN
    """

    def __init__(
        self,
        config: HAILPConfig,
        shared_ffn: SharedFFNPool,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.time_mix = RWKVTimeMixing(
            hidden_dim=config.hidden_dim,
            rank=config.low_rank_dim,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.hidden_dim)

        # Shared FFN — same nn.Module as other layers in this group
        self._shared_ffn_pool = shared_ffn
        self.ffn = shared_ffn.get_ffn_for_layer(layer_idx)

        self.adapter = LanguageAdapter(
            hidden_dim=config.hidden_dim,
            adapter_rank=config.adapter_rank,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through this block.

        Returns
        -------
        x : Tensor, shape (B, T, hidden_dim)
        h_state : Tensor, shape (B, hidden_dim)  — updated recurrent state
        """
        # Time mixing with residual
        attn_out, h_state = self.time_mix(self.ln1(x), h_state)
        x = x + attn_out

        # Shared FFN with residual
        x = x + self.ffn(self.ln2(x))

        # Language adapter
        x = self.adapter(x)

        return x, h_state


class HAILPModel(nn.Module):
    """H(AI)LP Model — recurrent architecture with fixed memory footprint.

    Replaces the growing KV cache of standard transformers with a fixed-size
    recurrent state, giving constant memory regardless of sequence length.

    Parameters
    ----------
    config : HAILPConfig
        Model configuration (loaded from configs/hailp.yaml).

    Example
    -------
    >>> cfg = HAILPConfig()
    >>> model = HAILPModel(cfg)
    >>> tokens = torch.randint(0, cfg.vocab_size, (2, 64))
    >>> logits, h = model(tokens)
    >>> logits.shape
    torch.Size([2, 64, 8000])
    >>> h[0].shape    # state for layer 0: fixed (B, hidden_dim) regardless of seq len
    torch.Size([2, 512])
    """

    def __init__(self, config: HAILPConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_drop = nn.Dropout(config.dropout)

        # Shared FFN pool — only `ffn_sharing_group_size` unique FFN blocks
        self.ffn_pool = SharedFFNPool(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.hidden_dim * 4,
            num_layers=config.layers,
            group_size=config.ffn_sharing_group_size,
        )

        # Build layers — each gets a reference to the shared pool
        self.blocks = nn.ModuleList([
            HAILPBlock(config, self.ffn_pool, layer_idx=i)
            for i in range(config.layers)
        ])

        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and "low_rank" not in name:
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def num_parameters(self) -> int:
        """Count unique trainable parameters (no double-counting shared weights)."""
        seen: set[int] = set()
        count = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                count += p.numel()
        return count

    @property
    def state_bytes(self) -> int:
        """Fixed state size in bytes (always same regardless of sequence length)."""
        return self.config.state_bytes

    def forward(
        self,
        input_ids: torch.Tensor,
        h_states: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        input_ids : Tensor, shape (B, T)
            Token indices in [0, vocab_size).
        h_states : list of Tensors or None
            Recurrent states, one per layer. Shape of each: (B, hidden_dim).
            If None, all states initialised to zeros.
            FIXED SIZE: regardless of T, each state is (B, hidden_dim).

        Returns
        -------
        logits : Tensor, shape (B, T, vocab_size)
        h_states : list[Tensor]
            Updated recurrent states for next call. Same fixed shapes.
        """
        b, t = input_ids.shape

        tok_emb = self.embed(input_ids)    # (b, t, hidden_dim)
        x = self.embed_drop(tok_emb)

        # Initialise states if not provided
        if h_states is None:
            h_states = [None] * len(self.blocks)

        new_h_states: list[torch.Tensor] = []
        for block, h in zip(self.blocks, h_states):
            x, h_new = block(x, h)
            new_h_states.append(h_new)

        x = self.ln_final(x)
        logits = self.head(x)             # (B, T, vocab_size)

        return logits, new_h_states

    @classmethod
    def from_config_file(cls, path: str | Path) -> HAILPModel:
        """Construct an H(AI)LP model from a YAML config file."""
        return cls(HAILPConfig.from_yaml(path))


