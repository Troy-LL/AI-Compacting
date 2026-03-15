"""Shared test fixtures for H(AI)LP tests."""

from __future__ import annotations

import pytest
import torch

from models.baseline_gpt import BaselineConfig, BaselineGPT
from models.hailp_model import HAILPConfig, HAILPModel


# ── Configs ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def baseline_config() -> BaselineConfig:
    """Full baseline config (6 layers, hidden=512)."""
    return BaselineConfig(
        layers=6,
        hidden_dim=512,
        attention_heads=8,
        ffn_expansion=4,
        vocab_size=8000,
        context_window=512,
        dropout=0.0,  # no dropout during testing
    )


@pytest.fixture(scope="session")
def hailp_config() -> HAILPConfig:
    """Full H(AI)LP config for parameter-count tests."""
    return HAILPConfig(
        layers=12,
        hidden_dim=512,
        vocab_size=8000,
        ffn_sharing_group_size=4,
        low_rank_dim=64,
        adapter_rank=32,
        dropout=0.0,
    )


# Legacy alias so existing test_lifelink.py fixtures still resolve
@pytest.fixture(scope="session")
def lifelink_config(hailp_config: HAILPConfig) -> HAILPConfig:  # noqa: F811
    """Alias for hailp_config (backwards compat)."""
    return hailp_config


@pytest.fixture(scope="session")
def tiny_baseline_config() -> BaselineConfig:
    """Tiny config for fast unit tests."""
    return BaselineConfig(
        layers=2,
        hidden_dim=64,
        attention_heads=4,
        ffn_expansion=2,
        vocab_size=100,
        context_window=32,
        dropout=0.0,
    )


@pytest.fixture(scope="session")
def tiny_hailp_config() -> HAILPConfig:
    """Tiny H(AI)LP config for fast unit tests."""
    return HAILPConfig(
        layers=4,
        hidden_dim=64,
        vocab_size=100,
        ffn_sharing_group_size=2,
        low_rank_dim=8,
        adapter_rank=4,
        dropout=0.0,
    )


# Legacy alias
@pytest.fixture(scope="session")
def tiny_lifelink_config(tiny_hailp_config: HAILPConfig) -> HAILPConfig:  # noqa: F811
    """Alias for tiny_hailp_config (backwards compat)."""
    return tiny_hailp_config


# ── Models ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_baseline_model(tiny_baseline_config: BaselineConfig) -> BaselineGPT:
    """Tiny baseline model (fast, for shape/functional tests)."""
    model = BaselineGPT(tiny_baseline_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_hailp_model(tiny_hailp_config: HAILPConfig) -> HAILPModel:
    """Tiny H(AI)LP model (fast, for shape/functional tests)."""
    model = HAILPModel(tiny_hailp_config)
    model.eval()
    return model


# Legacy alias so test_lifelink.py still works unchanged
@pytest.fixture(scope="session")
def tiny_lifelink_model(tiny_hailp_model: HAILPModel) -> HAILPModel:  # noqa: F811
    """Alias for tiny_hailp_model (backwards compat)."""
    return tiny_hailp_model


@pytest.fixture(scope="session")
def baseline_model(baseline_config: BaselineConfig) -> BaselineGPT:
    """Full baseline model (used for parameter count tests)."""
    model = BaselineGPT(baseline_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def hailp_model(hailp_config: HAILPConfig) -> HAILPModel:
    """Full H(AI)LP model (used for parameter count tests)."""
    model = HAILPModel(hailp_config)
    model.eval()
    return model


# Legacy alias
@pytest.fixture(scope="session")
def lifelink_model(hailp_model: HAILPModel) -> HAILPModel:  # noqa: F811
    """Alias for hailp_model (backwards compat)."""
    return hailp_model


# ── Helpers ────────────────────────────────────────────────────────────────────

@pytest.fixture
def device() -> torch.device:
    """Return CPU device (tests run CPU-only for CI compatibility)."""
    return torch.device("cpu")
