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

