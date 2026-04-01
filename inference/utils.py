"""Utility functions for inference."""

from __future__ import annotations

import hashlib
import re
from typing import Any


# Token estimation:
# - CJK chars: each char counts as a token
# - Word-like sequences
# - Any remaining non-space punctuation/symbol char
_TOKENS_RE = re.compile(
    r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]",
    flags=re.UNICODE,
)


def estimate_tokens(text: str) -> int:
    """Heuristic to estimate token count (no model tokenizer).

    In a real system, this would use the tokenizer. For H(AI)LP,
    we use a consistent regex-based heuristic that handles CJK and
    punctuation.
    """

    if not text:
        return 0
    return len(_TOKENS_RE.findall(text))


def call_model(
    model: Any,
    prompt: str,
    tokenizer: Any | None = None,
    sampler: Any | None = None,
) -> str:
    """Best-effort model adapter for various model interfaces.

    This ensures the pipeline can work with dummy test objects, scripted
    modules, and real PyTorch models.
    """

    if not prompt:
        return ""

    if model is None:
        return f"(model_inference) {prompt}"

    # 1. Try 'generate_text' (common in high-level wrappers)
    if hasattr(model, "generate_text") and callable(model.generate_text):
        return str(model.generate_text(prompt))

    # 2. Try 'respond' (internal project convention)
    if hasattr(model, "respond") and callable(model.respond):
        return str(model.respond(prompt))

    # 3. Try direct call (nn.Module or simple callable)
    if callable(model):
        try:
            return str(model(prompt))
        except TypeError:
            pass

    # 4. Try 'generate' (standard HF Transformers interface)
    if hasattr(model, "generate") and callable(model.generate):
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

    # Fallback to deterministic string for Phase 1-5 compatibility.
    return f"(model_inference) {prompt}"


def hash_document(document: str) -> str:
    """Create a stable SHA-256 cache key for a text document."""

    b = document.encode("utf-8", errors="ignore")
    return hashlib.sha256(b).hexdigest()
