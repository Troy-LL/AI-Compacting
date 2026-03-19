"""Scripted response router.

This router intercepts user queries *before* the model is called and tries to
handle them with:

1) a fast scripted lookup tier (survival-critical facts/guidance),
2) a fast deterministic computation tier (time, unit conversions),
3) model inference as a last resort.
"""

from __future__ import annotations

import logging
from typing import Any

from .fast_responder import RoutingMeta, RoutingResult, compute_response, lookup_response


def _model_inference(
    query: str,
    model: Any | None,
    tokenizer: Any | None,
    sampler: Any | None,
) -> str:
    """Best-effort model inference hook.

    The project will eventually integrate the router with the real model +
    tokenizer/sampler objects. For Phase 1 we keep this logic generic so tests
    can inject lightweight dummy objects.
    """

    prompt = query.strip()
    if not prompt:
        return ""

    if model is None:
        # Phase 1 fallback: still returns non-empty text.
        return f"(model_inference) {prompt}"

    # Common patterns for model adapters.
    if hasattr(model, "generate_text") and callable(getattr(model, "generate_text")):
        return str(model.generate_text(prompt))

    if hasattr(model, "respond") and callable(getattr(model, "respond")):
        return str(model.respond(prompt))

    if callable(model):
        try:
            return str(model(prompt))
        except TypeError:
            # Some callables require extra args; fall through to "generate".
            pass

    if hasattr(model, "generate") and callable(getattr(model, "generate")):
        generate = model.generate
        # Try a few argument styles, progressively decreasing assumptions.
        for kwargs in (
            {"tokenizer": tokenizer, "sampler": sampler},
            {"tokenizer": tokenizer},
            {},
        ):
            try:
                return str(generate(prompt, **kwargs))
            except TypeError:
                continue

    # If we couldn't adapt the model interface, return a deterministic string.
    return f"(model_inference) {prompt}"


def route_query(
    query: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    sampler: Any | None = None,
    *,
    logger: logging.Logger | None = None,
) -> RoutingResult:
    """Route a query through scripted tiers, then to model inference.

    Args:
        query: User input.
        model: Optional model object for the inference tier.
        tokenizer: Optional tokenizer object for the inference tier.
        sampler: Optional sampler/decoding configuration for the inference tier.
        logger: Optional logger (defaults to this module's logger).

    Returns:
        A tuple of (response_text, meta) where meta includes ``meta["path"]``.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    lookup = lookup_response(query)
    if lookup is not None:
        response, meta = lookup
        logger.info("route_query path=%s matched=%s", meta["path"], meta["matched"])
        return response, meta

    compute = compute_response(query)
    if compute is not None:
        response, meta = compute
        logger.info("route_query path=%s matched=%s", meta["path"], meta["matched"])
        return response, meta

    meta: RoutingMeta = {"path": "model_inference", "matched": "default"}
    response = _model_inference(query, model=model, tokenizer=tokenizer, sampler=sampler)
    logger.info("route_query path=%s matched=%s", meta["path"], meta["matched"])
    return response, meta

