"""Tests for model component modules.

Tests:
- test_low_rank_module        — shape, rank constraint, parameter count
- test_param_sharing_module   — same nn.Module referenced (not copied)
- test_adapter_injection      — adapter forward pass shape, residual identity init
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.components.adapter import LanguageAdapter
from models.components.low_rank import LowRankLinear
from models.components.param_sharing import SharedFFN, SharedFFNPool


# ── LowRankLinear ─────────────────────────────────────────────────────────────

class TestLowRankLinear:

    def test_output_shape(self) -> None:
        """LRL must produce the correct output shape."""
        lrl = LowRankLinear(512, 512, rank=64)
        x = torch.randn(2, 16, 512)
        out = lrl(x)
        assert out.shape == (2, 16, 512), f"Expected (2, 16, 512), got {out.shape}"

    def test_output_shape_non_square(self) -> None:
        """Works for non-square projections."""
        lrl = LowRankLinear(512, 256, rank=32)
        x = torch.randn(3, 8, 512)
        out = lrl(x)
        assert out.shape == (3, 8, 256)

    def test_rank_constraint_stored(self) -> None:
        """Rank attribute is accessible and correct."""
        lrl = LowRankLinear(512, 512, rank=64)
        assert lrl.effective_rank == 64

    def test_fewer_parameters_than_full_rank(self) -> None:
        """LRL should use fewer parameters than a full-rank Linear."""
        full_linear = nn.Linear(512, 512, bias=True)
        lrl = LowRankLinear(512, 512, rank=64, bias=True)

        full_params = sum(p.numel() for p in full_linear.parameters())
        lrl_params = lrl.num_parameters()

        assert lrl_params < full_params, (
            f"Low-rank {lrl_params} should be < full-rank {full_params}"
        )

    def test_parameter_count_formula(self) -> None:
        """Parameter count = rank*(in+out) + out (for bias)."""
        in_f, out_f, rank = 512, 512, 64
        lrl = LowRankLinear(in_f, out_f, rank=rank, bias=True)
        expected = rank * (in_f + out_f) + out_f
        assert lrl.num_parameters() == expected

    def test_invalid_rank_raises(self) -> None:
        """rank > min(in, out) must raise ValueError."""
        with pytest.raises(ValueError, match="rank"):
            LowRankLinear(64, 32, rank=33)

    def test_zero_rank_raises(self) -> None:
        """rank ≤ 0 must raise ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LowRankLinear(512, 512, rank=0)

    def test_compression_ratio(self) -> None:
        """compression_ratio should be > 1 for rank < min(in, out)."""
        lrl = LowRankLinear(512, 512, rank=64)
        assert lrl.compression_ratio() > 1.0

    def test_gradients_flow(self) -> None:
        """Gradients should flow through both V and U."""
        lrl = LowRankLinear(64, 64, rank=8)
        x = torch.randn(1, 4, 64, requires_grad=True)
        out = lrl(x)
        out.sum().backward()
        assert x.grad is not None
        assert lrl.V.weight.grad is not None
        assert lrl.U.weight.grad is not None


# ── SharedFFNPool ──────────────────────────────────────────────────────────────

class TestSharedFFNPool:

    def test_output_shape(self) -> None:
        """SharedFFN must return same shape as input."""
        pool = SharedFFNPool(hidden_dim=64, ffn_dim=256, num_layers=4, group_size=2)
        ffn = pool.get_ffn_for_layer(0)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_weight_sharing_is_identity(self) -> None:
        """Layers in the same group must reference THE SAME nn.Module — not a copy."""
        pool = SharedFFNPool(hidden_dim=64, ffn_dim=256, num_layers=12, group_size=4)

        # Layers 0, 4, 8 all map to block index 0 (layer_idx % group_size == 0)
        ffn_0 = pool.get_ffn_for_layer(0)
        ffn_4 = pool.get_ffn_for_layer(4)
        ffn_8 = pool.get_ffn_for_layer(8)

        assert ffn_0 is ffn_4, "Layers 0 and 4 should share the same FFN object"
        assert ffn_0 is ffn_8, "Layers 0 and 8 should share the same FFN object"

    def test_different_groups_are_different_modules(self) -> None:
        """Layers in different group positions have DIFFERENT nn.Module instances."""
        pool = SharedFFNPool(hidden_dim=64, ffn_dim=256, num_layers=8, group_size=4)

        ffn_0 = pool.get_ffn_for_layer(0)  # block index 0
        ffn_1 = pool.get_ffn_for_layer(1)  # block index 1

        assert ffn_0 is not ffn_1, "Layers at different group positions should differ"

    def test_num_unique_blocks(self) -> None:
        """Pool should contain exactly `group_size` unique blocks."""
        pool = SharedFFNPool(hidden_dim=64, ffn_dim=256, num_layers=12, group_size=4)
        assert len(pool.blocks) == 4

    def test_invalid_group_size_raises(self) -> None:
        """group_size=0 must raise ValueError."""
        with pytest.raises(ValueError):
            SharedFFNPool(hidden_dim=64, ffn_dim=256, num_layers=4, group_size=0)


# ── LanguageAdapter ────────────────────────────────────────────────────────────

class TestLanguageAdapter:

    def test_output_shape(self) -> None:
        """Adapter must produce the same shape as input (residual)."""
        adapter = LanguageAdapter(hidden_dim=512, adapter_rank=32)
        x = torch.randn(2, 16, 512)
        out = adapter(x)
        assert out.shape == x.shape

    def test_identity_at_init(self) -> None:
        """After weight init, adapter output ≈ input (residual + near-zero delta)."""
        adapter = LanguageAdapter(hidden_dim=64, adapter_rank=8)
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            out = adapter(x)
        # At init, up-projection is all zeros, so delta = 0 and out = x
        assert torch.allclose(out, x, atol=1e-5), (
            "Adapter should be near-identity at initialisation"
        )

    def test_gradients_flow(self) -> None:
        """Gradients should flow through the adapter."""
        adapter = LanguageAdapter(hidden_dim=64, adapter_rank=8)
        x = torch.randn(1, 4, 64, requires_grad=True)
        out = adapter(x)
        out.sum().backward()
        assert x.grad is not None

    def test_adapter_rank_too_large_raises(self) -> None:
        """adapter_rank > hidden_dim must raise ValueError."""
        with pytest.raises(ValueError, match="adapter_rank"):
            LanguageAdapter(hidden_dim=32, adapter_rank=64)

    def test_parameter_count(self) -> None:
        """Adapter params = 2*(hidden*rank + rank) + hidden (norm)."""
        hidden, rank = 512, 32
        adapter = LanguageAdapter(hidden_dim=hidden, adapter_rank=rank)
        params = adapter.num_parameters()
        # down: hidden*rank + rank, up: rank*hidden + hidden, norm: 2*hidden
        expected = (hidden * rank + rank) + (rank * hidden + hidden) + 2 * hidden
        assert params == expected, f"Expected {expected}, got {params}"
