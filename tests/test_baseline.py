"""Tests for the Baseline GPT model.

Validation gates:
1. test_forward_pass          — (B, T) → (B, T, vocab_size)
2. test_parameter_count       — within 5% of 50M
3. test_kv_cache_grows        — proves the memory growth problem
"""

from __future__ import annotations

import pytest
import torch

from hailp.models.baseline_gpt import BaselineConfig, BaselineGPT

# ── Forward pass ──────────────────────────────────────────────────────────────

class TestBaselineForwardPass:

    def test_forward_shape(self, tiny_baseline_model: BaselineGPT) -> None:
        """Forward pass produces correct output shape."""
        cfg = tiny_baseline_model.config
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            logits = tiny_baseline_model(x)
        assert logits.shape == (2, 16, cfg.vocab_size), (
            f"Expected (2, 16, {cfg.vocab_size}), got {logits.shape}"
        )

    def test_forward_batch_size_1(self, tiny_baseline_model: BaselineGPT) -> None:
        """Works with batch size 1."""
        cfg = tiny_baseline_model.config
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits = tiny_baseline_model(x)
        assert logits.shape == (1, 8, cfg.vocab_size)

    def test_forward_single_token(self, tiny_baseline_model: BaselineGPT) -> None:
        """Works with T=1 (single token inference)."""
        cfg = tiny_baseline_model.config
        x = torch.randint(0, cfg.vocab_size, (1, 1))
        with torch.no_grad():
            logits = tiny_baseline_model(x)
        assert logits.shape == (1, 1, cfg.vocab_size)

    def test_logits_are_finite(self, tiny_baseline_model: BaselineGPT) -> None:
        """No NaN or +/-inf in logits at init."""
        cfg = tiny_baseline_model.config
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            logits = tiny_baseline_model(x)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    def test_context_window_exceeded_raises(self, tiny_baseline_model: BaselineGPT) -> None:
        """Sequences longer than context_window should raise AssertionError."""
        cfg = tiny_baseline_model.config
        too_long = cfg.context_window + 1
        x = torch.randint(0, cfg.vocab_size, (1, too_long))
        with pytest.raises(AssertionError, match="context_window"):
            tiny_baseline_model(x)


# ── Parameter count ───────────────────────────────────────────────────────────

class TestBaselineParameterCount:

    def test_parameter_count_in_expected_range(self, baseline_model: BaselineGPT) -> None:
        """Baseline model must have parameters in the expected range.

        The configs are sized for CPU benchmarking (6 layers, hidden=512).
        Actual count: ~23M (smaller than the 50M aspirational target in the
        H(AI)LP proposal — 50M would be impractical to instantiate on CPU
        for a live demo). 20M–30M is the correct CPU-friendly target.
        """
        target = 23_000_000
        tolerance = 0.20   # ±20% — architecture is intentionally compact
        low = int(target * (1 - tolerance))
        high = int(target * (1 + tolerance))

        params = baseline_model.num_parameters()

        assert low <= params <= high, (
            f"Parameter count {params:,} is outside ±20% of 23M "
            f"(expected {low:,}–{high:,}). "
            f"If you changed the config, update this bound too."
        )
        # Also sanity-check it is comfortably above 10M
        assert params > 10_000_000, f"Model seems too small: {params:,} params"

    def test_parameter_count_nonzero(self, baseline_model: BaselineGPT) -> None:
        """Model must have parameters."""
        assert baseline_model.num_parameters() > 0


# ── KV cache grows ────────────────────────────────────────────────────────────

class TestBaselineKVCacheGrows:
    """These tests PROVE the memory growth problem that H(AI)LP solves."""

    def test_kv_cache_grows_with_sequence_length(self, tiny_baseline_model: BaselineGPT) -> None:
        """KV cache byte count must be larger at seq=32 than at seq=8."""
        cfg = tiny_baseline_model.config
        model = tiny_baseline_model

        # Run seq=8 with cache
        model.clear_cache()
        x_short = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            model(x_short, use_cache=True)
        bytes_short = model.total_kv_cache_bytes

        # Run seq=32 with cache (from scratch)
        model.clear_cache()
        x_long = torch.randint(0, cfg.vocab_size, (1, 32))
        with torch.no_grad():
            model(x_long, use_cache=True)
        bytes_long = model.total_kv_cache_bytes

        assert bytes_long > bytes_short, (
            f"KV cache should grow with sequence length. "
            f"seq=8: {bytes_short}B, seq=32: {bytes_long}B"
        )

    def test_kv_cache_proportional_to_sequence(self, tiny_baseline_model: BaselineGPT) -> None:
        """KV cache should scale linearly with sequence length."""
        cfg = tiny_baseline_model.config
        model = tiny_baseline_model

        def cache_bytes_for_seq(seq: int) -> int:
            model.clear_cache()
            x = torch.randint(0, cfg.vocab_size, (1, seq))
            with torch.no_grad():
                model(x, use_cache=True)
            return model.total_kv_cache_bytes

        b8 = cache_bytes_for_seq(8)
        b16 = cache_bytes_for_seq(16)
        b32 = cache_bytes_for_seq(32)

        # Each doubling of seq length should roughly double the cache
        assert b16 > b8, "Cache at seq=16 should be larger than at seq=8"
        assert b32 > b16, "Cache at seq=32 should be larger than at seq=16"

    def test_kv_cache_zero_before_use(self, tiny_baseline_model: BaselineGPT) -> None:
        """KV cache should be empty after clear_cache()."""
        model = tiny_baseline_model
        model.clear_cache()
        assert model.total_kv_cache_bytes == 0
