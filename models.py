from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DocumentUnit:
    """
    A provenance-preserving unit of text extracted from an uploaded file.

    Examples:
    - PDF: one unit per page (location_type='page', location_index=1..N)
    - DOCX: one unit per paragraph (location_type='paragraph', location_index=1..N)
    - PPTX: one unit per slide (location_type='slide', location_index=1..N)
    - CSV: one unit per row (location_type='row', location_index=1..N)
    - TXT/MD: one unit per line-batch (location_type='lines', location_index is the start line)
    """
    unit_id: int
    text: str
    source: str  # filename
    file_type: str  # pdf/docx/txt/md/csv/pptx
    location_type: str
    location: str
    location_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    """
    A chunk of text with full metadata for traceability.
    """
    chunk_id: int
    text: str
    source: str
    file_type: str
    location_type: str
    location: str
    location_index: Optional[int]
    unit_id: int
    word_span: str  # "start-end" in word offsets within the unit

    def citation_label(self) -> str:
        return f"{self.source} | {self.location}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # similarity score from FAISS

    def to_dict(self) -> Dict[str, Any]:
        d = {"score": self.score}
        d.update(self.chunk.to_dict())
        return d


@dataclass(frozen=True)
class SentenceAttribution:
    """
    Sentence-level traceability result.

    supported=True indicates the sentence is sufficiently grounded in evidence
    per similarity and optional entailment check.
    """
    sentence_id: int
    sentence: str
    best_chunk_id: Optional[int]
    best_similarity: float
    entailment_label: Optional[str] = None
    entailment_prob: Optional[float] = None
    supported: bool = False
    citation: Optional[str] = None
    evidence_snippet: Optional[str] = None
    candidates: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationEvent:
    """
    Log record for verification-effort measurement (human-study).
    """
    session_id: str
    event_type: str  # "verified"|"flagged"|"opened_evidence"
    timestamp_unix: float
    elapsed_seconds: Optional[float]
    source: str
    sentence_id: int
    sentence: str
    chunk_id: Optional[int]
    citation: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)