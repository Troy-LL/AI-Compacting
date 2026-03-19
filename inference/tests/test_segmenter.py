"""Tests for Phase 2 paragraph extraction and windowing."""

from __future__ import annotations

import pytest

from inference import segmenter


FIRST_AID_GUIDE = """First paragraph about water purification.

Follow these steps:
  - Boil water until you see a rolling boil.
  - Keep boiling for at least 1 minute.
  - Let water cool before drinking.

    Second paragraph about fire safety.

If clothing catches fire, stop, drop, and roll.

Note: if you can, seek medical help after any serious burns.
"""


WATER_PURIFICATION_GUIDE = """Water safety (quick reference).

 If the water is cloudy, let it settle and then filter it through cloth.

Slightly cloudy water: boil it. Clear water: boiling is still recommended if
you suspect contamination.

Conseil (français): faites bouillir l'eau 1 minute pour réduire les risques.
"""


LEGAL_RIGHTS_DOC = """Legal rights notice (plain language).

You have the right to be informed of the charges against you.
You have the right to a lawyer.

If you are a minor, additional protections may apply.

Section 1: confidentiality. Section 2: non-retaliation.
"""


def _make_long_paragraph() -> str:
    # Repeat a token-like phrase to exceed `segmenter.max_window_tokens`.
    return " ".join(["instruction"] * (segmenter.max_window_tokens + 80)) + "."


DOCUMENT = FIRST_AID_GUIDE

LONG_DOCUMENT = (
    "Paragraph A: short.\n\n"
    + "Paragraph B (very long): "
    + _make_long_paragraph()
    + "\n\n"
    + "Paragraph C: single-sentence only."
)

FIRST_AID_GUIDE_DOC = FIRST_AID_GUIDE
WATER_PURIFICATION_GUIDE_DOC = WATER_PURIFICATION_GUIDE
LEGAL_RIGHTS_DOC_TEXT = LEGAL_RIGHTS_DOC


def _make_atomic_near_max_paragraph() -> str:
    # Token estimator counts "word-ish" sequences as tokens, so each
    # repeated word contributes ~1 token.
    n_words = max(1, segmenter.max_window_tokens - 10)
    return " ".join(["instruction"] * n_words) + "."


ADVERSARIAL_DOCUMENT = (
    "Intro paragraph.\n\n"
    + "\n\n".join(_make_atomic_near_max_paragraph() for _ in range(40))
    + "\n\n"
    + "Final paragraph."
)


def test_paragraph_extraction() -> None:
    """Validation gate: basic paragraph extraction and ID assignment."""
    doc = """First paragraph about water.

    Second paragraph about fire.

    Third paragraph about shelter."""

    paras = segmenter.extract_paragraphs(doc)
    assert len(paras) == 3
    assert paras[0]["id"] == "p0"
    assert paras[1]["text"].strip().startswith("Second")


def test_window_token_limits() -> None:
    """Validation gate: token counts must not exceed the max per window."""
    paras = segmenter.extract_paragraphs(LONG_DOCUMENT)
    windows = segmenter.build_windows(paras)
    for window in windows:
        assert int(window["token_count"]) <= segmenter.max_window_tokens


def test_no_paragraphs_lost() -> None:
    """Validation gate: every paragraph ID appears somewhere in windows."""
    paras = segmenter.extract_paragraphs(DOCUMENT)
    windows = segmenter.build_windows(paras)
    all_para_ids = {p["id"] for p in paras}
    windowed_ids = {p["id"] for w in windows for p in w["paragraphs"]}
    assert all_para_ids == windowed_ids


def test_real_document_first_aid() -> None:
    """First-aid guide should segment into multiple paragraphs/windows."""
    paras = segmenter.extract_paragraphs(FIRST_AID_GUIDE)
    assert len(paras) >= 3
    windows = segmenter.build_windows(paras)
    assert len(windows) >= 1
    assert sum(len(w["paragraphs"]) for w in windows) >= len(paras)


def test_real_document_water_purification() -> None:
    """Water purification guide should be handled even with multilingual text."""
    paras = segmenter.extract_paragraphs(WATER_PURIFICATION_GUIDE)
    assert len(paras) >= 2
    windows = segmenter.build_windows(paras)
    assert len(windows) >= 1


def test_real_document_legal_rights() -> None:
    """Legal rights notices should segment reliably for downstream context."""
    paras = segmenter.extract_paragraphs(LEGAL_RIGHTS_DOC)
    assert len(paras) >= 3
    windows = segmenter.build_windows(paras)
    # Token budget is deterministic; this ensures we don't produce empty windows.
    assert all(len(w["paragraphs"]) > 0 for w in windows)


def test_boundary_response_parsing() -> None:
    """Validation gate: parse boundary responses into paragraph ids."""
    paras = [
        {"id": "p0", "text": "a", "token_count": 1},
        {"id": "p1", "text": "b", "token_count": 1},
        {"id": "p2", "text": "c", "token_count": 1},
        {"id": "p3", "text": "d", "token_count": 1},
        {"id": "p4", "text": "e", "token_count": 1},
    ]

    assert segmenter._parse_boundary_response("p3", paras) == "p3"
    assert (
        segmenter._parse_boundary_response("The shift is at p3.", paras) == "p3"
    )
    assert segmenter._parse_boundary_response("paragraph p3", paras) == "p3"
    assert segmenter._parse_boundary_response("none", paras) is None
    assert segmenter._parse_boundary_response("no shift", paras) is None


def test_boundary_token_cost() -> None:
    """Validation gate: boundary detection output should be very short."""
    # Window with 5 paragraphs → midpoint is p2 (len//2).
    window = {
        "token_count": 5,
        "paragraphs": [
            {"id": "p0", "text": "a", "token_count": 1},
            {"id": "p1", "text": "b", "token_count": 1},
            {"id": "p2", "text": "c", "token_count": 1},
            {"id": "p3", "text": "d", "token_count": 1},
            {"id": "p4", "text": "e", "token_count": 1},
        ],
    }
    _boundary, meta = segmenter.find_boundary_with_cost(window)
    assert meta["tokens_used"] < 20


def test_unparseable_fallback() -> None:
    """Validation gate: unparseable responses should not crash."""
    paras = [
        {"id": "p0", "text": "a", "token_count": 1},
        {"id": "p1", "text": "b", "token_count": 1},
        {"id": "p2", "text": "c", "token_count": 1},
    ]
    result = segmenter._parse_boundary_response("asdfjkl;", paras)
    assert result is None


def test_full_document_segmentation() -> None:
    """Validation gate: segmentation respects budgets + covers most words."""
    segments = segmenter.segment(FIRST_AID_GUIDE_DOC)
    assert segments is not None
    for seg in segments:
        assert int(seg["token_count"]) <= segmenter.max_window_tokens

    original_words = set(FIRST_AID_GUIDE_DOC.split())
    segmented_words = set(" ".join(str(s["text"]) for s in segments).split())
    coverage = len(original_words & segmented_words) / max(1, len(original_words))
    assert coverage > 0.95


def test_no_infinite_loop() -> None:
    """Validation gate: segmentation terminates promptly."""
    import time
    import signal

    def run_segment() -> list[dict[str, object]]:
        return segmenter.segment(ADVERSARIAL_DOCUMENT)

    # Windows compatibility: SIGALRM is not available in all environments.
    if hasattr(signal, "SIGALRM") and hasattr(signal, "alarm"):
        try:
            def timeout_handler(signum: int, frame: object) -> None:  # noqa: ARG001
                raise TimeoutError("Segmentation took too long")

            signal.signal(signal.SIGALRM, timeout_handler)  # type: ignore[arg-type]
            signal.alarm(30)  # type: ignore[attr-defined]
            try:
                segments = run_segment()
                assert segments is not None
            finally:
                signal.alarm(0)  # type: ignore[attr-defined]
            return
        except Exception:
            # Fall back to wall-clock timing in case the environment doesn't
            # support SIGALRM reliably.
            pass

    start = time.time()
    segments = run_segment()
    elapsed = time.time() - start
    assert segments is not None
    assert elapsed < 30

