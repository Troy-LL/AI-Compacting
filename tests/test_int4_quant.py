"""Tests for INT4 pack/unpack helpers used in quantisation."""

from __future__ import annotations

import torch

from hailp.benchmarks.quantize_int4 import (
    load_hailp_int4,
    pack_int4_tight,
    pack_model,
    save_hailp_int4,
    should_quantize_int4,
    unpack_int4_tight,
)
from hailp.models.hailp_model import HAILPConfig, HAILPModel


def test_pack_unpack_roundtrip() -> None:
    tensor = torch.randn(512, 512)
    packed, scale, zp = pack_int4_tight(tensor)

    # Two 4‑bit values per byte (rounding up).
    assert packed.numel() == (tensor.numel() + 1) // 2

    recovered = unpack_int4_tight(packed, scale, zp, tensor.shape)
    assert recovered.shape == tensor.shape

    max_error = (tensor - recovered).abs().max().item()
    # For 4‑bit per‑tensor quantisation on random data, a max error below
    # ~0.5 is acceptable for this storage‑oriented demo.
    assert max_error < 0.5


def test_tensor_routing() -> None:
    assert not should_quantize_int4("layers.0.bias", torch.randn(512))
    assert not should_quantize_int4("norm.weight", torch.randn(512))
    assert not should_quantize_int4("weight", torch.randn(32, 32))
    assert should_quantize_int4("layers.0.weight", torch.randn(512, 512))


def test_pack_model_coverage() -> None:
    state_dict = {
        "layers.0.weight": torch.randn(512, 512),
        "layers.0.bias": torch.randn(512),
        "norm.weight": torch.randn(512),
        "small.weight": torch.randn(16, 16),
    }

    quantized, kept_fp16 = pack_model(state_dict)
    original_names = set(state_dict.keys())
    quantized_names = set(quantized.keys())
    fp16_names = set(kept_fp16.keys())

    assert original_names == quantized_names | fp16_names


def test_full_save_load_roundtrip(tmp_path) -> None:
    cfg = HAILPConfig(
        layers=2,
        hidden_dim=64,
        vocab_size=1000,
        ffn_sharing_group_size=2,
        low_rank_dim=8,
        adapter_rank=4,
        dropout=0.0,
    )
    model = HAILPModel(cfg)

    out_prefix = tmp_path / "hailp_test"
    npz_path = save_hailp_int4(model, out_prefix)
    size_mb = npz_path.stat().st_size / (1024 ** 2)
    assert size_mb < 12, f"Expected <12MB, got {size_mb:.1f}MB"

    fresh_model = HAILPModel(cfg)
    load_hailp_int4(out_prefix, fresh_model)

    test_input = torch.randint(0, cfg.vocab_size, (1, 64))
    with torch.no_grad():
        original_out, _ = model(test_input)
        loaded_out, _ = fresh_model(test_input)

    max_error = (original_out - loaded_out).abs().max().item()
    assert max_error < 0.5

