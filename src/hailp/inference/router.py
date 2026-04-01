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
from .utils import call_model


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
    response = call_model(model, query.strip(), tokenizer=tokenizer, sampler=sampler)
    logger.info("route_query path=%s matched=%s", meta["path"], meta["matched"])
    return response, meta

