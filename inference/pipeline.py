"""Unified inference pipeline.

Phase 5 connects:
- Phase 1 scripted router (zero-to-low compute)
- Phase 2-4 paragraph segmentation + boundary detection
- Model inference fallback
- Telemetry (path, token estimate, latency)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from . import segmenter
from .fast_responder import compute_response, lookup_response
from .telemetry import Telemetry


def _estimate_tokens(text: str) -> int:
    """Estimate tokens using the segmenter's heuristic."""

    # Reuse the same heuristic so Phase 2/4 budgets and telemetry stay consistent.
    return int(segmenter._estimate_token_count(text))  # type: ignore[attr-defined]


def _call_model(model: Any, prompt: str, tokenizer: Any | None, sampler: Any | None) -> str:
    """Best-effort model adapter for dummy models and real integrations."""

    if not prompt:
        return ""

    if model is None:
        return f"(model_inference) {prompt}"

    if hasattr(model, "generate_text") and callable(getattr(model, "generate_text")):
        return str(model.generate_text(prompt))

    if hasattr(model, "respond") and callable(getattr(model, "respond")):
        return str(model.respond(prompt))

    if callable(model):
        try:
            return str(model(prompt))
        except TypeError:
            pass

    if hasattr(model, "generate") and callable(getattr(model, "generate")):
        generate = model.generate
        for kwargs in (
            {"tokenizer": tokenizer, "sampler": sampler},
            {"tokenizer": tokenizer},
            {},
        ):
            try:
                return str(generate(prompt, **kwargs))
            except TypeError:
                continue

    return f"(model_inference) {prompt}"


def _hash_document(document: str) -> str:
    """Create a stable cache key for a document."""

    b = document.encode("utf-8", errors="ignore")
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline tuning knobs."""

    overlap_paragraphs: int = 1
    # Maximum number of segments to include in the context string.
    # This keeps prompts bounded for real model integrations.
    max_segments_in_context: int = 8


class HailpInferencePipeline:
    """Unify scripted routing, segmentation, and model inference."""

    def __init__(
        self,
        model: Any | None,
        tokenizer: Any | None = None,
        sampler: Any | None = None,
        *,
        config: PipelineConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._sampler = sampler
        self._config = config or PipelineConfig()
        self._logger = logger or logging.getLogger(__name__)

        # Simple in-memory cache for segmentation outputs.
        self._segment_cache: dict[str, list[dict[str, object]]] = {}

    def _segment_context(self, context_document: str) -> list[dict[str, object]]:
        """Segment context with a small in-memory cache."""

        key = _hash_document(context_document)
        cached = self._segment_cache.get(key)
        if cached is not None:
            return cached

        segments = segmenter.segment(
            context_document,
            overlap_paragraphs=self._config.overlap_paragraphs,
            # For Phase 5 tests we keep boundary detection deterministic and
            # avoid calling the model inside the segmentation loop.
            model=None,
            tokenizer=None,
            sampler=None,
        )
        self._segment_cache[key] = segments
        return segments

    def answer_with_meta(
        self, query: str, *, context_document: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Answer a query and return telemetry metadata.

        Args:
            query: User query.
            context_document: Optional text used for segmentation-assisted
                answering.

        Returns:
            (answer_text, meta) where meta includes:
            - ``path``: scripted_lookup|segmented_inference|model_inference
            - ``tokens_used``: heuristic token estimate (0 for scripted tiers)
            - ``latency_ms``: wall-clock latency in milliseconds
        """

        t0 = time.perf_counter()
        try:
            lookup = lookup_response(query)
            if lookup is not None:
                response, meta = lookup
                telemetry = Telemetry(
                    path=meta["path"],
                    tokens_used=0,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
                self._logger.info(
                    "pipeline path=%s tokens_used=%s",
                    telemetry.path,
                    telemetry.tokens_used,
                )
                return response, {
                    "path": telemetry.path,
                    "tokens_used": telemetry.tokens_used,
                    "latency_ms": telemetry.latency_ms,
                }

            compute = compute_response(query)
            if compute is not None:
                response, meta = compute
                telemetry = Telemetry(
                    path=meta["path"],
                    tokens_used=0,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
                self._logger.info(
                    "pipeline path=%s tokens_used=%s",
                    telemetry.path,
                    telemetry.tokens_used,
                )
                return response, {
                    "path": telemetry.path,
                    "tokens_used": telemetry.tokens_used,
                    "latency_ms": telemetry.latency_ms,
                }

            if context_document is not None:
                segments = self._segment_context(context_document)
                used_segments = segments[: self._config.max_segments_in_context]
                context_text = " ".join(str(s.get("text", "")).strip() for s in used_segments).strip()

                prompt = (
                    f"User query: {query}\n\n"
                    f"Context:\n{context_text}\n\n"
                    "Answer based on the context. If unsure, say so briefly."
                )
                response = _call_model(
                    self._model, prompt, tokenizer=self._tokenizer, sampler=self._sampler
                )
                tokens_used = _estimate_tokens(prompt)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._logger.info(
                    "pipeline path=segmented_inference tokens_used=%s",
                    tokens_used,
                )
                return response, {
                    "path": "segmented_inference",
                    "tokens_used": tokens_used,
                    "latency_ms": latency_ms,
                }

            # No context: direct model inference tier.
            response = _call_model(
                self._model, query.strip(), tokenizer=self._tokenizer, sampler=self._sampler
            )
            tokens_used = _estimate_tokens(query.strip())
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._logger.info("pipeline path=model_inference tokens_used=%s", tokens_used)
            return response, {
                "path": "model_inference",
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
            }
        except Exception:
            # Fail safe: ensure tests and demos don't crash the whole pipeline.
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._logger.exception("pipeline failed")
            return "", {"path": "model_inference", "tokens_used": 0, "latency_ms": latency_ms}

