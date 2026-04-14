from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import AppConfig, get_config
from models import RetrievedChunk, SentenceAttribution
from text_utils import split_sentences, truncate_chars


@dataclass
class NLIResult:
    label: str
    prob: float


class NLIJudge:
    """
    Optional NLI entailment checker.

    If transformers or the model is unavailable, instantiate with enabled=False.
    """
    def __init__(self, model_name: str, enabled: bool = True):
        self.enabled = enabled
        self.model_name = model_name
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline
        self._pipe = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            return_all_scores=True,
            truncation=True,
        )

    def entailment(self, premise: str, hypothesis: str) -> Optional[NLIResult]:
        if not self.enabled:
            return None
        try:
            self._load()
            outputs = self._pipe((premise, hypothesis))
            scores = outputs[0]
            best = max(scores, key=lambda x: x.get("score", 0.0))
            label = str(best.get("label", "")).lower()
            prob = float(best.get("score", 0.0))
            if "entail" in label:
                norm_label = "entailment"
            elif "contrad" in label:
                norm_label = "contradiction"
            else:
                norm_label = "neutral"
            return NLIResult(label=norm_label, prob=prob)
        except Exception:
            return None


def attribute_sentences(
    sentences: List[str],
    retrieved: List[RetrievedChunk],
    embed_fn,
    cfg: AppConfig | None = None,
    nli: Optional[NLIJudge] = None,
    top_m: int = 3,
) -> List[SentenceAttribution]:
    """
    Map each sentence to the most supportive retrieved chunk(s) and compute grounding.

    Report mapping:
    - Sentence-level grounding score (semantic similarity + optional entailment)
    - Evidence snippets and page-level citations per sentence
    """
    cfg = cfg or get_config()
    if not sentences:
        return []
    if not retrieved:
        return [
            SentenceAttribution(
                sentence_id=i,
                sentence=s,
                best_chunk_id=None,
                best_similarity=0.0,
                supported=False,
                citation=None,
                evidence_snippet=None,
                candidates=[],
            )
            for i, s in enumerate(sentences)
        ]

    chunk_texts = [r.chunk.text for r in retrieved]
    chunk_ids = [r.chunk.chunk_id for r in retrieved]

    sent_emb = embed_fn(sentences)
    chunk_emb = embed_fn(chunk_texts)

    sim = np.matmul(sent_emb, chunk_emb.T)  # cosine if embeddings normalized
    attributions: List[SentenceAttribution] = []

    for i, s in enumerate(sentences):
        row = sim[i]
        order = np.argsort(-row)
        best_j = int(order[0])
        best_score = float(row[best_j])

        best_chunk = retrieved[best_j].chunk
        citation = best_chunk.citation_label()
        snippet = truncate_chars(best_chunk.text, 320)

        nli_res = None
        if nli is not None and nli.enabled:
            nli_res = nli.entailment(premise=best_chunk.text, hypothesis=s)

        supported = best_score >= cfg.SUPPORT_SIM_THRESHOLD
        if nli_res is not None:
            if nli_res.label == "contradiction":
                supported = False
            elif nli_res.label == "entailment":
                supported = supported and (nli_res.prob >= cfg.ENTAILMENT_THRESHOLD)

        candidates = []
        for j in order[: min(top_m, len(order))]:
            j = int(j)
            c = retrieved[j].chunk
            candidates.append(
                {
                    "chunk_id": c.chunk_id,
                    "similarity": float(row[j]),
                    "citation": c.citation_label(),
                    "snippet": truncate_chars(c.text, 200),
                }
            )

        attributions.append(
            SentenceAttribution(
                sentence_id=i,
                sentence=s,
                best_chunk_id=best_chunk.chunk_id,
                best_similarity=best_score,
                entailment_label=nli_res.label if nli_res else None,
                entailment_prob=nli_res.prob if nli_res else None,
                supported=supported,
                citation=citation,
                evidence_snippet=snippet,
                candidates=candidates,
            )
        )

    return attributions


def hallucination_rate(attributions: List[SentenceAttribution]) -> float:
    """
    Sentence-level hallucination proxy:
    fraction of sentences not supported by retrieved evidence.
    """
    if not attributions:
        return 0.0
    unsupported = sum(1 for a in attributions if not a.supported)
    return unsupported / len(attributions)


def mean_grounding(attributions: List[SentenceAttribution]) -> float:
    if not attributions:
        return 0.0
    return float(np.mean([a.best_similarity for a in attributions]))


def split_text_to_sentences(text: str) -> List[str]:
    return split_sentences(text)