"""Tests for the scripted response router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hailp.inference.router import route_query


@dataclass(frozen=True)
class DummyModel:
    """A lightweight model adapter for router tests."""

    def generate_text(self, prompt: str) -> str:
        return f"MODEL_INFERENCE: {prompt}"


class DummyTokenizer:
    """Tokenizer stub (unused by DummyModel)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


class DummySampler:
    """Sampler stub (unused by DummyModel)."""

    def sample(self, text: str) -> str:
        return text


@pytest.fixture()
def model() -> DummyModel:
    """Provide a dummy model for inference-tier tests."""

    return DummyModel()


@pytest.fixture()
def tokenizer() -> DummyTokenizer:
    """Provide a dummy tokenizer for inference-tier tests."""

    return DummyTokenizer()


@pytest.fixture()
def sampler() -> DummySampler:
    """Provide a dummy sampler for inference-tier tests."""

    return DummySampler()


def test_scripted_routing() -> None:
    """Validation gate: ensure tier selection matches expectations."""

    assert route_query("SOS")[1]["path"] == "scripted_lookup"
    assert route_query("what time is it")[1]["path"] == "scripted_compute"
    assert route_query("explain photosynthesis")[1]["path"] == "model_inference"


def test_nothing_dropped(model: Any, tokenizer: Any, sampler: Any) -> None:
    """Validation gate: every input should receive a non-empty response."""

    queries = [
        "SOS",
        "boil water",
        "explain photosynthesis",
        "celsius to fahrenheit 100",
        "emergency number",
    ]
    for q in queries:
        result, _meta = route_query(q, model, tokenizer, sampler)
        assert result is not None
        assert len(result) > 0

