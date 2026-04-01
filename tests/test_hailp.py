"""Tests for the H(AI)LP model.

Validation gates:
1. test_forward_pass               — (B, T) → (B, T, vocab_size)
2. test_parameter_count            — within ±20% of 18M target
3. test_parameter_sharing_active   — layers share the same nn.Module
4. test_state_is_fixed_size        — seq=64 state == seq=512 state (THE KEY CLAIM)
5. test_low_rank_projections       — attention projections are rank-decomposed
"""

from __future__ import annotations

import torch

from models.hailp_model import HAILPModel

# ── Forward pass ──────────────────────────────────────────────────────────────

class TestHAILPForwardPass:

    def test_forward_shape(self, tiny_hailp_model: HAILPModel) -> None:
        """Forward pass produces (B, T, vocab_size) logits."""
        cfg = tiny_hailp_model.config
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            logits, h_states = tiny_hailp_model(x)
        assert logits.shape == (2, 16, cfg.vocab_size), (
            f"Expected (2, 16, {cfg.vocab_size}), got {logits.shape}"
        )

    def test_h_states_returned(self, tiny_hailp_model: HAILPModel) -> None:
        """Forward pass returns one h_state per layer."""
        cfg = tiny_hailp_model.config
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            _, h_states = tiny_hailp_model(x)
        assert len(h_states) == cfg.layers, (
            f"Expected {cfg.layers} states, got {len(h_states)}"
        )

    def test_h_state_shapes(self, tiny_hailp_model: HAILPModel) -> None:
        """Each h_state has shape (B, hidden_dim)."""
        cfg = tiny_hailp_model.config
        B, T = 3, 12
        x = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            _, h_states = tiny_hailp_model(x)
        for i, h in enumerate(h_states):
            assert h.shape == (B, cfg.hidden_dim), (
                f"Layer {i} state: expected ({B}, {cfg.hidden_dim}), got {h.shape}"
            )

    def test_logits_are_finite(self, tiny_hailp_model: HAILPModel) -> None:
        """No NaN or Inf in logits at initialisation."""
        cfg = tiny_hailp_model.config
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            logits, _ = tiny_hailp_model(x)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    def test_stateful_inference(self, tiny_hailp_model: HAILPModel) -> None:
        """Can pass h_states back in for stateful (streaming) inference."""
        cfg = tiny_hailp_model.config
        x1 = torch.randint(0, cfg.vocab_size, (1, 8))
        x2 = torch.randint(0, cfg.vocab_size, (1, 8))

        with torch.no_grad():
            _, h1 = tiny_hailp_model(x1)
            logits2, h2 = tiny_hailp_model(x2, h_states=h1)

        assert logits2.shape == (1, 8, cfg.vocab_size)
        assert len(h2) == cfg.layers


# ── Parameter count ───────────────────────────────────────────────────────────

class TestHAILPParameterCount:

    def test_parameter_count_in_expected_range(self, hailp_model: HAILPModel) -> None:
        """H(AI)LP unique parameter count must be in the expected range.

        With FFN sharing (group_size=4, 12 layers → 3 unique FFN blocks)
        and low-rank projections (rank=64), unique params are ~18M.
        This is deliberately fewer than the Baseline's ~23M despite having
        2× more layers — that's the whole point of parameter sharing.
        """
        target = 18_000_000
        tolerance = 0.20   # ±20% — generous given architecture flexibility
        low = int(target * (1 - tolerance))
        high = int(target * (1 + tolerance))

        params = hailp_model.num_parameters()

        assert low <= params <= high, (
            f"Parameter count {params:,} is outside ±20% of 18M "
            f"(expected {low:,}–{high:,}). "
            f"If you changed the config, update this bound too."
        )
        assert params > 5_000_000, f"Model seems too small: {params:,} params"

    def test_parameter_count_nonzero(self, hailp_model: HAILPModel) -> None:
        """Model must have parameters."""
        assert hailp_model.num_parameters() > 0


# ── Parameter sharing ─────────────────────────────────────────────────────────

class TestHAILPParameterSharing:

    def test_parameter_sharing_active(self, tiny_hailp_model: HAILPModel) -> None:
        """Layers with the same group position share the SAME nn.Module for FFN."""
        cfg = tiny_hailp_model.config
        group_size = cfg.ffn_sharing_group_size

        # Layers 0 and group_size should share block index 0
        ffn_layer0 = tiny_hailp_model.blocks[0].ffn
        ffn_layer_gs = tiny_hailp_model.blocks[group_size].ffn

        assert ffn_layer0 is ffn_layer_gs, (
            f"Layers 0 and {group_size} should share the same FFN object, "
            f"but got different instances (the ffn_pool is not being used correctly)"
        )

    def test_shared_ffn_param_data_ptr(self, tiny_hailp_model: HAILPModel) -> None:
        """Shared FFN weights have identical data pointers (same storage)."""
        cfg = tiny_hailp_model.config
        group_size = cfg.ffn_sharing_group_size

        w0 = tiny_hailp_model.blocks[0].ffn.fc1.weight
        w_gs = tiny_hailp_model.blocks[group_size].ffn.fc1.weight

        assert w0.data_ptr() == w_gs.data_ptr(), (
            "Shared FFN weights should point to the same memory"
        )

    def test_total_unique_ffn_blocks(self, tiny_hailp_model: HAILPModel) -> None:
        """Number of unique FFN nn.Module instances = group_size (not num_layers)."""
        cfg = tiny_hailp_model.config
        ffn_set = {id(block.ffn) for block in tiny_hailp_model.blocks}
        assert len(ffn_set) == cfg.ffn_sharing_group_size, (
            f"Expected {cfg.ffn_sharing_group_size} unique FFN blocks, "
            f"got {len(ffn_set)}"
        )


# ── Fixed state size — THE KEY CLAIM ─────────────────────────────────────────

class TestHAILPFixedState:
    """Proves that H(AI)LP's memory footprint is constant regardless of seq len."""

    def test_state_is_fixed_size(self, tiny_hailp_model: HAILPModel) -> None:
        """h_state shapes must be IDENTICAL for seq=8 and seq=32."""
        cfg = tiny_hailp_model.config
        B = 1

        x_short = torch.randint(0, cfg.vocab_size, (B, 8))
        x_long = torch.randint(0, cfg.vocab_size, (B, 32))

        with torch.no_grad():
            _, h_short = tiny_hailp_model(x_short)
            _, h_long = tiny_hailp_model(x_long)

        for i, (hs, hl) in enumerate(zip(h_short, h_long)):
            assert hs.shape == hl.shape, (
                f"Layer {i}: h_state shape differs between seq=8 ({hs.shape}) "
                f"and seq=32 ({hl.shape}). State must be FIXED SIZE!"
            )

    def test_state_byte_count_independent_of_sequence(
        self,
        tiny_hailp_model: HAILPModel,
    ) -> None:
        """Total bytes of all h_states is the same for any sequence length."""
        cfg = tiny_hailp_model.config
        B = 1

        def state_bytes(seq: int) -> int:
            x = torch.randint(0, cfg.vocab_size, (B, seq))
            with torch.no_grad():
                _, h_states = tiny_hailp_model(x)
            return sum(h.numel() * h.element_size() for h in h_states)

        b8 = state_bytes(8)
        b32 = state_bytes(32)

        assert b8 == b32, (
            f"State bytes should be seq-independent: seq=8 → {b8}B, seq=32 → {b32}B"
        )


# ── Low-rank projections ──────────────────────────────────────────────────────

class TestHAILPLowRankProjections:

    def test_attention_projections_are_low_rank(
        self,
        tiny_hailp_model: HAILPModel,
    ) -> None:
        """Time-mixing projections must be LowRankLinear instances."""
        from models.components.low_rank import LowRankLinear

        for i, block in enumerate(tiny_hailp_model.blocks):
            assert isinstance(block.time_mix.w_q, LowRankLinear), (
                f"Block {i}: w_q is not LowRankLinear"
            )
            assert isinstance(block.time_mix.w_k, LowRankLinear), (
                f"Block {i}: w_k is not LowRankLinear"
            )
            assert isinstance(block.time_mix.w_v, LowRankLinear), (
                f"Block {i}: w_v is not LowRankLinear"
            )

    def test_low_rank_dim_respected(self, tiny_hailp_model: HAILPModel) -> None:
        """LowRankLinear modules use the configured rank."""
        cfg = tiny_hailp_model.config
        for i, block in enumerate(tiny_hailp_model.blocks):
            assert block.time_mix.w_q.effective_rank == cfg.low_rank_dim, (
                f"Block {i}: w_q rank is {block.time_mix.w_q.effective_rank}, "
                f"expected {cfg.low_rank_dim}"
            )
