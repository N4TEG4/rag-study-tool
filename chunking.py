from __future__ import annotations

from typing import List

from config import AppConfig, get_config
from models import Chunk, DocumentUnit


def chunk_units(units: List[DocumentUnit], cfg: AppConfig | None = None) -> List[Chunk]:
    """
    Chunk document units into word-window chunks, preserving metadata.

    Report mapping:
    - Chunk metadata includes chunk_id, source, page/location, and word_span.
    """
    cfg = cfg or get_config()
    chunks: List[Chunk] = []
    chunk_id = 0

    for u in units:
        words = [w for w in (u.text or "").split() if w.strip()]
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + cfg.CHUNK_SIZE_WORDS, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=u.source,
                        file_type=u.file_type,
                        location_type=u.location_type,
                        location=u.location,
                        location_index=u.location_index,
                        unit_id=u.unit_id,
                        word_span=f"{start}-{end}",
                    )
                )
                chunk_id += 1

            if end >= len(words):
                break
            start = max(0, end - cfg.CHUNK_OVERLAP_WORDS)

            if chunk_id >= cfg.MAX_TOTAL_CHUNKS:
                return chunks

    return chunks