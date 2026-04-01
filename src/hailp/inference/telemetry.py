"""Lightweight telemetry utilities for the inference pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Telemetry:
    """Structured telemetry payload returned alongside model outputs."""

    path: str
    tokens_used: int
    latency_ms: float


def now_ms() -> float:
    """Return a high-resolution timestamp in milliseconds."""

    return time.perf_counter() * 1000.0

