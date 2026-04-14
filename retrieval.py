from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import AppConfig, get_config
from models import Chunk, RetrievedChunk


@dataclass
class VectorIndex:
    """
    sentence-transformers + FAISS vector store.

    Uses IndexFlatIP over L2-normalized embeddings, so inner product corresponds
    to cosine similarity for normalized vectors.
    """
    cfg: AppConfig
    model: SentenceTransformer
    index: Optional[faiss.Index] = None
    chunks: Optional[List[Chunk]] = None
    embeddings: Optional[np.ndarray] = None  # cached embeddings for traceability

    @classmethod
    def create(cls, cfg: AppConfig | None = None) -> "VectorIndex":
        cfg = cfg or get_config()
        model = SentenceTransformer(cfg.EMBEDDING_MODEL)
        return cls(cfg=cfg, model=model)

    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        return emb

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            self.index = None
            self.chunks = []
            self.embeddings = None
            return

        embs = self.embed([c.text for c in chunks])
        dim = embs.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(embs)

        self.index = idx
        self.chunks = chunks
        self.embeddings = embs

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if self.index is None or not self.chunks:
            return []

        top_k = top_k or self.cfg.TOP_K_RETRIEVAL
        q = self.embed([query])
        scores, ids = self.index.search(q, top_k)

        results: List[RetrievedChunk] = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[int(i)], score=float(score)))
        return results

    def topk_for_summary_query(self, focus: str | None = None) -> List[RetrievedChunk]:
        """
        Default retrieval query for summarisation when user provides no question.
        """
        q = focus or (
            "Summarise the material for a university student. Extract key concepts, definitions, "
            "methods, findings, and any dates/deadlines. Keep to the provided material."
        )
        return self.search(q, top_k=self.cfg.TOP_K_RETRIEVAL)