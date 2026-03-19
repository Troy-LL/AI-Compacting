"""Text segmentation for H(AI)LP.

Phase 2: paragraph extraction and context window building.

Goals:
- Extract stable paragraph IDs from a document.
- Normalize whitespace so edge cases don't create "phantom" paragraphs.
- Build windows that respect a maximum token budget.
- Handle paragraphs longer than the window limit by splitting them across
  windows without losing their original paragraph ID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# A conservative token budget for a "window" in Phase 2.
# This is an estimation (not model tokenizer tokens) and is only used to
# decide how to chunk text deterministically.
max_window_tokens: int = 200


_SPLIT_PARAGRAPHS_RE = re.compile(r"(?:\r?\n)\s*(?:\r?\n)+", flags=re.MULTILINE)
_MULTISPACE_RE = re.compile(r"\s+")

# Token estimation:
# - CJK chars: each char counts as a token
# - Word-like sequences
# - Any remaining non-space punctuation/symbol char
_TOKENS_RE = re.compile(
    r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]",
    flags=re.UNICODE,
)


def _normalize_paragraph_text(text: str) -> str:
    # Collapse inconsistent whitespace (spaces/tabs/newlines) inside a paragraph.
    return _MULTISPACE_RE.sub(" ", text).strip()


def _estimate_token_count(text: str) -> int:
    """Estimate token count using a fast heuristic (no model tokenizer)."""
    if not text:
        return 0
    return len(_TOKENS_RE.findall(text))


def _tokenize(text: str) -> list[str]:
    """Tokenize for counting + chunk splitting (heuristic, model-agnostic)."""
    if not text:
        return []
    return _TOKENS_RE.findall(text)


def _chunk_tokens(tokens: list[str], max_tokens: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_count = 0
    for tok in tokens:
        # Each element in `tokens` is treated as one token by the heuristic.
        if current_count + 1 > max_tokens and current:
            chunks.append(current)
            current = []
            current_count = 0
        current.append(tok)
        current_count += 1
    if current:
        chunks.append(current)
    return chunks


def _join_tokens(tokens: Iterable[str]) -> str:
    """Join heuristic tokens back into readable-ish chunk text."""

    # Insert spaces between consecutive "word-ish" tokens; avoid before punctuation.
    out: list[str] = []
    prev_was_word = False
    for tok in tokens:
        is_word = bool(re.match(r"^[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?$", tok)) or bool(
            re.match(r"^[\u4e00-\u9fff]$", tok)
        )
        is_punct = not is_word

        if not out:
            out.append(tok)
        else:
            if prev_was_word and not is_punct and not tok.isspace():
                # This branch is mostly for word-word concatenation; we keep it simple.
                out.append(" " + tok)
            else:
                # For punctuation/symbols, don't prepend extra whitespace.
                if prev_was_word and is_word:
                    out.append(" " + tok)
                else:
                    out.append(tok)

        prev_was_word = is_word
    return "".join(out).strip()


@dataclass(frozen=True)
class _Paragraph:
    id: str
    text: str


def extract_paragraphs(document: str) -> list[dict[str, str]]:
    """Extract normalized paragraphs and assign stable IDs.

    Paragraphs are split on blank lines. Leading/trailing whitespace and
    inconsistent internal whitespace are normalized.

    Args:
        document: Input document text.

    Returns:
        A list of dicts: ``{"id": <pid>, "text": <paragraph_text>}``.
    """

    if document is None:
        return []

    normalized_doc = document.replace("\r\n", "\n").strip()
    if not normalized_doc:
        return []

    raw_paragraphs = _SPLIT_PARAGRAPHS_RE.split(normalized_doc)
    paragraphs: list[_Paragraph] = []

    for idx, raw in enumerate(raw_paragraphs):
        text = _normalize_paragraph_text(raw)
        if not text:
            continue
        paragraphs.append(_Paragraph(id=f"p{len(paragraphs)}", text=text))

    # Convert to the dict shape expected by Phase 2 tests.
    return [{"id": p.id, "text": p.text} for p in paragraphs]


def build_windows(paragraphs: list[dict[str, str]]) -> list[dict[str, object]]:
    """Group paragraphs into token-limited windows.

    If a paragraph's estimated token count exceeds ``max_window_tokens``,
    it is split into chunks across windows while preserving the original
    paragraph ID.

    Args:
        paragraphs: Output from :func:`extract_paragraphs`.

    Returns:
        List of windows, each shaped like::

            {
              "token_count": <int>,
              "paragraphs": [{"id": ..., "text": ..., "token_count": ...}, ...]
            }
    """

    # Defensive: accept empty input.
    if not paragraphs:
        return []

    # Internal representation
    parsed = [_Paragraph(id=p["id"], text=p["text"]) for p in paragraphs]

    windows: list[dict[str, object]] = []
    current_paragraphs: list[dict[str, object]] = []
    current_tokens = 0

    def flush_window() -> None:
        nonlocal current_paragraphs, current_tokens
        if not current_paragraphs:
            return
        windows.append({"token_count": current_tokens, "paragraphs": current_paragraphs})
        current_paragraphs = []
        current_tokens = 0

    for p in parsed:
        p_tokens = _estimate_token_count(p.text)

        if p_tokens <= max_window_tokens:
            # If paragraph fits, pack greedily; otherwise start new window.
            if current_tokens + p_tokens > max_window_tokens:
                flush_window()
            current_paragraphs.append(
                {"id": p.id, "text": p.text, "token_count": p_tokens}
            )
            current_tokens += p_tokens
            continue

        # Paragraph is too long: split by heuristic tokens.
        tokens = _tokenize(p.text)
        chunks = _chunk_tokens(tokens, max_window_tokens)

        for chunk_idx, chunk_tokens in enumerate(chunks):
            chunk_text = _join_tokens(chunk_tokens)
            chunk_token_count = _estimate_token_count(chunk_text)

            # Safety: in case join/token estimate disagree, enforce the budget.
            if chunk_token_count > max_window_tokens:
                # Fallback: hard-trim by tokens.
                chunk_text = _join_tokens(chunk_tokens[:max_window_tokens])
                chunk_token_count = _estimate_token_count(chunk_text)

            if current_tokens + chunk_token_count > max_window_tokens:
                flush_window()

            current_paragraphs.append(
                {
                    "id": p.id,
                    "text": chunk_text,
                    "token_count": chunk_token_count,
                    # Not used by Phase 2 tests, but helps debugging later.
                    "chunk_index": chunk_idx,
                }
            )
            current_tokens += chunk_token_count

    flush_window()
    return windows

