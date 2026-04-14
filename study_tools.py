from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from text_utils import split_sentences


def _tfidf_keyphrases(text: str, top_k: int = 12) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=500)
    X = vec.fit_transform(sents)
    terms = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    out: List[str] = []
    for t, _ in ranked:
        if len(t) >= 3:
            out.append(t)
        if len(out) >= top_k:
            break
    return out


def generate_flashcards(evidence_text: str, max_cards: int = 8) -> List[Tuple[str, str]]:
    """
    Algorithmic flashcards:
    - Extract TF-IDF keyphrases from evidence
    - Find a sentence containing the keyphrase as an answer
    """
    keyphrases = _tfidf_keyphrases(evidence_text, top_k=max_cards * 2)
    sentences = split_sentences(evidence_text)
    cards: List[Tuple[str, str]] = []
    used_answers = set()

    for kp in keyphrases:
        answer = ""
        for s in sentences:
            if kp.lower() in s.lower() and s not in used_answers:
                answer = s
                used_answers.add(s)
                break
        if answer:
            q = f"What does '{kp}' refer to in these materials?"
            cards.append((q, answer))
        if len(cards) >= max_cards:
            break
    return cards


_DATE_RE = re.compile(
    r"\b("
    r"(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}"
    r"|"
    r"\d{4}"
    r")\b",
    re.IGNORECASE,
)


def generate_timeline(evidence_text: str, max_items: int = 8) -> List[str]:
    """
    Algorithmic timeline: extract sentences containing year/month-date patterns.
    """
    events: List[str] = []
    seen = set()
    for s in split_sentences(evidence_text):
        for m in _DATE_RE.findall(s):
            key = m.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            events.append(f"{m}: {s}")
            if len(events) >= max_items:
                return events
    return events


_ACTION_MARKERS = [
    "must", "should", "need to", "required", "important", "ensure", "verify",
    "implement", "compare", "measure", "evaluate", "analyse", "test", "submit",
    "deadline", "due", "complete",
]


def generate_checklist(evidence_text: str, max_items: int = 10) -> List[str]:
    """
    Algorithmic checklist: sentences containing action markers.
    """
    items: List[str] = []
    for s in split_sentences(evidence_text):
        low = s.lower()
        if any(m in low for m in _ACTION_MARKERS):
            items.append(s)
        if len(items) >= max_items:
            break
    if not items:
        items = split_sentences(evidence_text)[:max_items]
    return items[:max_items]


def generate_study_plan(
    evidence_text: str,
    days: int = 5,
    sessions_per_day: int = 2,
) -> List[Dict[str, str]]:
    """
    Algorithmic study plan: distribute key topics over sessions.
    """
    days = max(1, min(days, 30))
    sessions_per_day = max(1, min(sessions_per_day, 4))

    topics = _tfidf_keyphrases(evidence_text, top_k=days * sessions_per_day * 2)
    if not topics:
        topics = ["Review core concepts", "Re-read evidence snippets", "Self-test with checklist"]

    plan: List[Dict[str, str]] = []
    idx = 0

    for d in range(1, days + 1):
        for s in range(1, sessions_per_day + 1):
            chunk = topics[idx : idx + 3]
            idx += 3
            if not chunk:
                chunk = topics[:3]
            plan.append(
                {
                    "day": f"Day {d}",
                    "session": f"Session {s}",
                    "focus": ", ".join(chunk),
                    "task": "Summarise these points in your own words and create 3 self-test questions.",
                }
            )

    return plan


def textrank_summary(text: str, max_sentences: int = 8) -> List[str]:
    """
    Extractive TextRank (graph-based ranking of sentences).
    Baseline for comparison against LLM-only and RAG.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= max_sentences:
        return sentences

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)
    sim = (X * X.T).toarray()

    g = nx.Graph()
    g.add_nodes_from(range(len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            w = float(sim[i, j])
            if w > 0:
                g.add_edge(i, j, weight=w)

    scores = nx.pagerank(g, weight="weight")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picked = sorted([i for i, _ in ranked[:max_sentences]])
    return [sentences[i] for i in picked]