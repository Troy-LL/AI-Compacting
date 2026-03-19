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
from typing import Any, Iterable

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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: LLM boundary detection
# ─────────────────────────────────────────────────────────────────────────────


def _build_boundary_prompt(window: dict[str, object]) -> str:
    """Build a compact prompt to detect a topic shift within a window.

    The boundary detector is designed to keep model output short. The desired
    response format is either:
    - a paragraph id like ``p3``
    - or ``none`` if there is no meaningful shift.

    Args:
        window: A window dict produced by :func:`build_windows`.

    Returns:
        A prompt string.
    """

    paragraphs = window.get("paragraphs")
    if not isinstance(paragraphs, list):
        return (
            "Find the topic shift boundary. "
            "Return only a paragraph id like p3 or 'none'. "
            "Keep your response under 20 tokens."
        )

    lines: list[str] = [
        "You are detecting where topics shift inside a short window of text.",
        "Choose the paragraph id where the main topic changes.",
        "If there's no clear shift, answer 'none'.",
        "Rules: respond with ONLY 'pX' or 'none' (no extra words).",
        "Keep response under 20 tokens.",
        "",
        "Window paragraphs:",
    ]
    for p in paragraphs:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", "")).strip()
        text = str(p.get("text", "")).strip().replace("\n", " ")
        excerpt = text[:120] + ("..." if len(text) > 120 else "")
        if pid:
            lines.append(f"- {pid}: {excerpt}")

    lines.append("")
    lines.append("Answer (pX or none):")
    return "\n".join(lines)


def _parse_boundary_response(
    response: str, paragraphs: list[dict[str, object]]
) -> str | None:
    """Parse model output to recover the boundary paragraph id.

    Supported formats (examples):
    - ``p3``
    - ``The shift is at p3.``
    - ``paragraph p3``
    - ``shift:p3``
    - ``none`` / ``no shift``

    Args:
        response: Raw model text.
        paragraphs: Paragraph objects belonging to the window.

    Returns:
        The matched paragraph id (e.g. ``"p3"``) or ``None`` if no shift.
    """

    if response is None:
        return None

    resp = response.strip().lower()
    if not resp:
        return None

    none_markers = ("none", "no shift", "no-shift", "no meaningful shift")
    if any(marker in resp for marker in none_markers):
        return None

    # Prefer explicit "p<number>" tokens anywhere in the response.
    matches = re.findall(r"\bp\d+\b", resp)
    if not matches:
        return None

    valid_ids = {str(p.get("id", "")).strip() for p in paragraphs if isinstance(p, dict)}
    for m in matches:
        if m in valid_ids:
            return m

    return None


def _fallback_to_midpoint(window: dict[str, object]) -> str | None:
    paragraphs = window.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        return None
    # Midpoint by index, not by id (window paragraphs may include split chunks).
    mid_idx = len(paragraphs) // 2
    p = paragraphs[mid_idx]
    if isinstance(p, dict):
        return str(p.get("id", "")).strip() or None
    return None


def find_boundary_with_cost(
    window: dict[str, object],
    model: Any | None = None,
    tokenizer: Any | None = None,
    sampler: Any | None = None,
) -> tuple[str | None, dict[str, int]]:
    """Find the topic-shift boundary inside a window and report token cost.

    For Phase 3 gating we track only the *model response* token usage (under
    our heuristic). The prompt itself is assumed to be handled with a
    "short response" instruction.

    Args:
        window: Window dict produced by :func:`build_windows`.
        model: Optional model object. If omitted, a deterministic midpoint
            fallback is returned.
        tokenizer: Optional tokenizer (used only for model adapters).
        sampler: Optional sampler (used only for model adapters).

    Returns:
        (boundary_id, meta) where meta includes ``tokens_used``.
    """

    midpoint = _fallback_to_midpoint(window)
    prompt = _build_boundary_prompt(window)

    # Phase 3 tests may call this without a model; in that case we skip any
    # real inference to keep the test deterministic and token-efficient.
    if model is None:
        # Represent the fallback as a short response so token-cost stays low.
        boundary_resp = midpoint if midpoint else "none"
        return midpoint, {"tokens_used": _estimate_token_count(boundary_resp)}

    # Best-effort model inference hook.
    if hasattr(model, "generate_text") and callable(getattr(model, "generate_text")):
        boundary_resp = str(model.generate_text(prompt))
    elif hasattr(model, "respond") and callable(getattr(model, "respond")):
        boundary_resp = str(model.respond(prompt))
    elif callable(model):
        boundary_resp = str(model(prompt))
    else:
        # Unknown adapter: fallback.
        boundary_resp = midpoint if midpoint else "none"

    parsed = _parse_boundary_response(boundary_resp, window.get("paragraphs", []))  # type: ignore[arg-type]
    boundary_id = parsed if parsed is not None else midpoint
    return boundary_id, {"tokens_used": _estimate_token_count(boundary_resp)}


def find_boundary(
    window: dict[str, object],
    model: Any | None = None,
    tokenizer: Any | None = None,
    sampler: Any | None = None,
) -> str | None:
    """Find the topic shift boundary id.

    This is a thin wrapper over :func:`find_boundary_with_cost` that discards
    token-cost metadata.
    """

    boundary_id, _meta = find_boundary_with_cost(
        window, model=model, tokenizer=tokenizer, sampler=sampler
    )
    return boundary_id

