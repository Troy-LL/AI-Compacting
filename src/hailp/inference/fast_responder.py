"""Fast scripted routing helpers.

This module implements two "zero-to-low compute" tiers that can answer
certain classes of prompts without calling the model:

1) :func:`lookup_response` — a small lookup table for survival-critical
   guidance (SOS, emergency numbers, boiling water, basic first aid).
2) :func:`compute_response` — deterministic computations (e.g. Celsius to
   Fahrenheit) and lightweight utilities (e.g. current time).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from re import Pattern
from typing import TypedDict


class RoutingMeta(TypedDict):
    """Metadata describing which routing tier matched the query."""

    path: str
    matched: str


RoutingResult = tuple[str, RoutingMeta]


def _normalize_query(query: str) -> str:
    """Normalize a query for robust pattern matching."""

    # Keep punctuation because SOS variants may use dots/spaces (e.g. "S.O.S").
    normalized = query.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _format_number(x: float) -> str:
    """Format a number for human output (no trailing .0)."""

    if x.is_integer():
        return str(int(x))
    # Keep precision reasonable for small "mental math" style answers.
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s


@dataclass(frozen=True)
class _LookupRule:
    name: str
    patterns: tuple[Pattern[str], ...]
    response: str

    def matches(self, normalized_query: str) -> bool:
        return any(p.search(normalized_query) is not None for p in self.patterns)


_LOOKUP_RULES: tuple[_LookupRule, ...] = (
    _LookupRule(
        name="sos_signal",
        patterns=(
            re.compile(r"\bsos\b"),
            # Handles common "S O S" and "S.O.S" variants.
            re.compile(r"s[\W_]*o[\W_]*s"),
        ),
        response=(
            "SOS is a distress signal (Morse code: ... --- ...).\n"
            "If you need help: signal for attention repeatedly, "
            "use bright light/sound if available, and provide location/details."
        ),
    ),
    _LookupRule(
        name="emergency_number",
        patterns=(
            re.compile(r"\bemergency\s*(number|numbers)?\b"),
            re.compile(r"\b(call|dial)\s*(911|112|999|000)\b"),
            re.compile(r"\b(911|112|999|000)\b"),
        ),
        response=(
            "Emergency numbers:\n"
            "- 911 (US/Canada)\n"
            "- 112 (many countries in Europe and beyond)\n"
            "- 999 (UK, varies by region)\n"
            "- 000 (Australia)\n"
            "If unsure, call your local emergency services."
        ),
    ),
    _LookupRule(
        name="boil_water",
        patterns=(
            re.compile(r"\bboil\s*water\b"),
            re.compile(r"\boil\s*water\b"),
            re.compile(r"\bwater\s+purification\b"),
            re.compile(r"\bpurify\s*water\b"),
        ),
        response=(
            "Boil-water guidance (general):\n"
            "- Bring water to a rolling boil.\n"
            "- Boil for at least 1 minute (3 minutes at higher altitudes if advised).\n"
            "- Let cool before drinking."
        ),
    ),
    _LookupRule(
        name="basic_first_aid_bleeding",
        patterns=(re.compile(r"\bbleed(ing)?\b"), re.compile(r"\bsevere\s+bleeding\b")),
        response=(
            "Basic first aid for heavy bleeding (general):\n"
            "- Apply firm direct pressure to the wound.\n"
            "- If possible, elevate the injured area.\n"
            "- Keep pressure on until professional help arrives."
        ),
    ),
    _LookupRule(
        name="basic_first_aid_burn",
        patterns=(re.compile(r"\bburn(ing)?\b"), re.compile(r"\bthermal\s+burn\b")),
        response=(
            "Basic first aid for burns (general):\n"
            "- Cool the burn under cool running water for ~20 minutes (if safe to do so).\n"
            "- Remove rings/belts/watch near the area (before swelling).\n"
            "- Cover loosely with a clean, non-stick dressing."
        ),
    ),
)


def lookup_response(query: str) -> RoutingResult | None:
    """Return a scripted lookup response, if any.

    Args:
        query: User input.

    Returns:
        A tuple of (response_text, meta) if a rule matches, otherwise ``None``.
    """

    if not query:
        return None

    normalized = _normalize_query(query)
    for rule in _LOOKUP_RULES:
        if rule.matches(normalized):
            return (
                rule.response,
                {
                    "path": "scripted_lookup",
                    "matched": rule.name,
                },
            )
    return None


def compute_response(query: str) -> RoutingResult | None:
    """Return a deterministic scripted computation response, if any.

    Args:
        query: User input.

    Returns:
        A tuple of (response_text, meta) if a computation matches, otherwise
        ``None``.
    """

    if not query:
        return None

    normalized = _normalize_query(query)

    # Current time utility.
    if re.search(r"\bwhat\s+time\s+is\s+it\b", normalized) or re.search(
        r"\bcurrent\s+time\b", normalized
    ):
        now = datetime.now()
        time_str = now.strftime("%I:%M %p").lstrip("0")
        return (
            f"Current local time is {time_str}.",
            {"path": "scripted_compute", "matched": "local_time"},
        )

    # Celsius -> Fahrenheit conversion.
    m = re.search(
        r"\bcelsius\s*to\s*fahrenheit\s*(-?\d+(?:\.\d+)?)\b",
        normalized,
    )
    if m is not None:
        c = float(m.group(1))
        f = c * 9.0 / 5.0 + 32.0
        return (
            f"{_format_number(c)} C = {_format_number(f)} F.",
            {"path": "scripted_compute", "matched": "c_to_f"},
        )

    # Fahrenheit -> Celsius conversion.
    m = re.search(
        r"\bfahrenheit\s*to\s*celsius\s*(-?\d+(?:\.\d+)?)\b",
        normalized,
    )
    if m is not None:
        f = float(m.group(1))
        c = (f - 32.0) * 5.0 / 9.0
        return (
            f"{_format_number(f)} F = {_format_number(c)} C.",
            {"path": "scripted_compute", "matched": "f_to_c"},
        )

    return None

