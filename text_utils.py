from __future__ import annotations

import re
from typing import List


_WS = re.compile(r"\s+")


def normalise_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter.
    Keeps the project dependency-light
    """
    if not text:
        return []
    text = normalise_whitespace(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def count_words(text: str) -> int:
    if not text:
        return 0
    return len([w for w in text.split() if w.strip()])


def truncate_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def safe_join_text(blocks: List[str], sep: str = "\n") -> str:
    blocks = [b for b in blocks if b and b.strip()]
    return sep.join(blocks)