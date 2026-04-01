"""Baseline GPT — standard full-attention transformer at ~50M parameters.

Used as the CONTROL model in H(AI)LP. Demonstrates the KV-cache memory growth
problem that H(AI)LP RWKV is designed to solve.

Architecture
------------
- 6 layers of causal self-attention (full O(n²))
- hidden_dim = 512, 8 heads, FFN = 4 × hidden_dim = 2048
- vocab_size = 8000, context_window = 512
- KV cache stored as list of (K, V) tensors that grows with sequence length

The KV-cache memory grows as:
    2 × layers × heads × seq_len × head_dim × sizeof(float32)

At seq=64:   2 × 6 × 8 × 64 × 64 × 4 bytes  =  1.5 MB
At seq=512:  2 × 6 × 8 × 512 × 64 × 4 bytes  = 12.6 MB  (8× larger)

This is the problem. H(AI)LP RWKV replaces this with a fixed 512-byte state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class BaselineConfig:
    """Configuration for the Baseline GPT model."""

    layers: int = 6
    hidden_dim: int = 512
    attention_heads: int = 8
    ffn_expansion: int = 4
    vocab_size: int = 8000
    context_window: int = 512
    dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: str | Path) -> BaselineConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            layers=cfg.get("layers", 6),
            hidden_dim=cfg.get("hidden_dim", 512),
            attention_heads=cfg.get("attention_heads", 8),
            ffn_expansion=cfg.get("ffn_expansion", 4),
            vocab_size=cfg.get("vocab_size", 8000),
            context_window=cfg.get("context_window", 512),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.attention_heads

    @property
    def ffn_dim(self) -> int:
        return self.hidden_dim * self.ffn_expansion


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention with optional KV cache.

    On first forward pass, creates a KV cache. On subsequent calls with
    `use_cache=True`, appends the new K,V to the cache and attends over
    the full cached history.

    The cache is stored as two tensors of shape (B, H, T_so_far, head_dim).
    Memory grows linearly with T_so_far — THE PROBLEM we're measuring.
    """

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.num_heads = config.attention_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        # Fused QKV projection + output projection
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask (registered as buffer — not a parameter)
        mask = torch.tril(torch.ones(config.context_window, config.context_window))
        self.register_buffer("mask", mask.view(1, 1, config.context_window, config.context_window))

        # KV cache (None until first forward, then grows with sequence)
        self._kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def clear_cache(self) -> None:
        """Reset the KV cache (call between independent sequences)."""
        self._kv_cache = None

    @property
    def cache_bytes(self) -> int:
        """Return current KV cache size in bytes."""
        if self._kv_cache is None:
            return 0
        k, v = self._kv_cache
        return (k.numel() + v.numel()) * k.element_size()

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, T, hidden_dim)
        use_cache : bool
            If True, append new K,V to the running cache.

        Returns
        -------
        Tensor, shape (B, T, hidden_dim)
        """
        b, t, c = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to multi-head: (b, H, t, head_dim)
        def reshape(t_in: torch.Tensor) -> torch.Tensor:
            return t_in.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # KV cache: concatenate past and present
        if use_cache and self._kv_cache is not None:
            past_k, past_v = self._kv_cache
            k = torch.cat([past_k, k], dim=2)  # (b, H, t_total, head_dim)
            v = torch.cat([past_v, v], dim=2)

        if use_cache:
            self._kv_cache = (k.detach(), v.detach())

        t_total = k.shape[2]

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale         # (b, H, t, t_total)

        # Apply causal mask
        causal_mask = self.mask[:, :, :t, :t_total]
        attn = attn.masked_fill(causal_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate values
        out = attn @ v                                     # (b, H, t, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(out)


class GPTBlock(nn.Module):
    """Standard transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), use_cache=use_cache)
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineGPT(nn.Module):
    """Baseline GPT — the control model for H(AI)LP.

    Standard causal transformer. Its KV cache grows O(n) with sequence length.

    Parameters
    ----------
    config : BaselineConfig
        Model configuration (loaded from configs/baseline.yaml).

    Example
    -------
    >>> cfg = BaselineConfig()
    >>> model = BaselineGPT(cfg)
    >>> tokens = torch.randint(0, cfg.vocab_size, (2, 64))
    >>> logits = model(tokens)
    >>> logits.shape
    torch.Size([2, 64, 8000])
    """

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(config.context_window, config.hidden_dim)
        self.embed_drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.layers)])
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying: embed and head share the same weight matrix
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def clear_cache(self) -> None:
        """Reset all KV caches. Call between independent sequences."""
        for block in self.blocks:
            block.attn.clear_cache()

    @property
    def total_kv_cache_bytes(self) -> int:
        """Total KV cache memory in bytes across all layers."""
        return sum(block.attn.cache_bytes for block in self.blocks)

    def num_parameters(self) -> int:
        """Count trainable parameters (excluding weight-tied duplicates)."""
        # head.weight is tied to embed.weight — don't double count
        params = set()
        for p in self.parameters():
            params.add(p.data_ptr())
        return sum(p.numel() for p in self.parameters() if p.data_ptr() in params)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids : Tensor, shape (B, T)
            Token indices in [0, vocab_size).
        use_cache : bool
            If True, maintain KV cache across calls.

        Returns
        -------
        logits : Tensor, shape (B, T, vocab_size)
        """
        b, t = input_ids.shape
        assert t <= self.config.context_window, (
            f"Sequence length {t} exceeds context_window {self.config.context_window}"
        )

        tok_emb = self.embed(input_ids)                     # (b, t, hidden_dim)
        pos = torch.arange(t, device=input_ids.device)
        pos_emb = self.pos_embed(pos)                       # (t, hidden_dim)
        x = self.embed_drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, use_cache=use_cache)

        x = self.ln_final(x)
        logits = self.head(x)                               # (B, T, vocab_size)
        return logits

    @classmethod
    def from_config_file(cls, path: str | Path) -> BaselineGPT:
        """Construct a BaselineGPT from a YAML config file."""
        return cls(BaselineConfig.from_yaml(path))
