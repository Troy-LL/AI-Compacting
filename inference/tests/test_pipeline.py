"""Tests for Phase 5 pipeline integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from inference.pipeline import HailpInferencePipeline
from inference.tests.test_segmenter import FIRST_AID_GUIDE as first_aid_guide


@dataclass(frozen=True)
class DummyModel:
    """A lightweight model adapter for pipeline tests."""

    def generate_text(self, prompt: str) -> str:
        return f"MODEL_INFERENCE_RESPONSE: {prompt}"


@dataclass(frozen=True)
class DummyTokenizer:
    """Tokenizer stub."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


@pytest.fixture()
def model() -> DummyModel:
    return DummyModel()


@pytest.fixture()
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


def test_pipeline_end_to_end(model: Any, tokenizer: Any) -> None:
    """Validation gate: end-to-end tier selection + response meta."""
    pipeline = HailpInferencePipeline(model, tokenizer)

    result, meta = pipeline.answer_with_meta("SOS signal")
    assert result is not None
    assert meta["tokens_used"] == 0
    assert meta["path"] == "scripted_lookup"

    result, meta = pipeline.answer_with_meta(
        "How do I treat a wound?", context_document=first_aid_guide
    )
    assert result is not None
    assert meta["path"] == "segmented_inference"

    result, meta = pipeline.answer_with_meta("What should I do if I am lost?")
    assert result is not None
    assert meta["path"] == "model_inference"


def test_telemetry_complete(model: Any, tokenizer: Any) -> None:
    """Validation gate: telemetry payload must include required keys."""
    pipeline = HailpInferencePipeline(model, tokenizer)
    _, meta = pipeline.answer_with_meta("SOS signal")
    assert "path" in meta
    assert "tokens_used" in meta
    assert "latency_ms" in meta

