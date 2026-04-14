from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI

from config import AppConfig, get_config
from models import RetrievedChunk
from text_utils import split_sentences, truncate_chars


def _chunks_to_context(retrieved: List[RetrievedChunk], max_chars: int = 12000) -> str:
    """
    Render retrieved chunks as a prompt context block with provenance information.
    """
    blocks: List[str] = []
    for r in retrieved:
        c = r.chunk
        header = f"[CHUNK {c.chunk_id} | {c.source} | {c.location} | span {c.word_span}]"
        blocks.append(header + "\n" + c.text)

    ctx = "\n\n".join(blocks)
    return truncate_chars(ctx, max_chars=max_chars)


def _openai_client(cfg: AppConfig) -> OpenAI:
    """
    Create an OpenAI client using either hosted OpenAI or an OpenAI-compatible endpoint.
    """
    kwargs: Dict[str, Any] = {}

    if cfg.OPENAI_API_KEY:
        kwargs["api_key"] = cfg.OPENAI_API_KEY

    if cfg.OPENAI_BASE_URL:
        kwargs["base_url"] = cfg.OPENAI_BASE_URL

    return OpenAI(**kwargs)


def _try_structured_json(
    client: OpenAI,
    cfg: AppConfig,
    messages: List[Dict[str, Any]],
    schema_name: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Try Structured Outputs first, then fall back to JSON mode.
    """
    if cfg.OPENAI_USE_STRUCTURED_OUTPUTS:
        try:
            completion = client.chat.completions.create(
                model=cfg.OPENAI_MODEL,
                messages=messages,
                temperature=cfg.OPENAI_TEMPERATURE,
                max_tokens=cfg.OPENAI_MAX_OUTPUT_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            content = completion.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            pass

    completion = client.chat.completions.create(
        model=cfg.OPENAI_MODEL,
        messages=messages,
        temperature=cfg.OPENAI_TEMPERATURE,
        max_tokens=cfg.OPENAI_MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content or "{}"

    try:
        return json.loads(content)
    except Exception:
        return {"text": content}


@dataclass
class LLMGenerator:
    """
    Generator supporting:
    - baseline LLM-only summarisation
    - retrieval-conditioned RAG summarisation
    - retrieval-conditioned RAG question answering
    """

    cfg: AppConfig

    @classmethod
    def create(cls, cfg: AppConfig | None = None) -> "LLMGenerator":
        cfg = cfg or get_config()
        return cls(cfg=cfg)

    def generate_rag_summary(
        self,
        retrieved: List[RetrievedChunk],
        focus: str | None = None,
    ) -> Dict[str, Any]:
        """
        Generate a retrieval-conditioned summary using only retrieved chunks.

        Returns:
            {
                "sentences": [...],
                "summary": "..."
            }
        """
        client = _openai_client(self.cfg)
        context = _chunks_to_context(retrieved)

        instruction = (
            "You are an academic summarisation assistant. "
            "STRICT RULES:\n"
            "1) Use ONLY the provided chunks as your information source.\n"
            "2) Do NOT add outside knowledge.\n"
            "3) Write in British English.\n"
            "4) If unsupported, omit the claim.\n"
        )

        user_focus = (focus or "Produce a general academic summary.").strip()

        prompt = (
            f"{instruction}\n"
            f"Write approximately {self.cfg.SUMMARY_SENTENCE_TARGET} sentences.\n\n"
            f"FOCUS:\n{user_focus}\n\n"
            f"CHUNKS:\n{context}\n\n"
            "Return JSON with:\n"
            "- sentences: array of strings\n"
            "- summary: string\n"
        )

        schema = {
            "type": "object",
            "properties": {
                "sentences": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "summary": {"type": "string"},
            },
            "required": ["sentences", "summary"],
            "additionalProperties": False,
        }

        messages = [
            {"role": "system", "content": "Follow instructions exactly."},
            {"role": "user", "content": prompt},
        ]

        data = _try_structured_json(client, self.cfg, messages, "rag_summary", schema)

        sentences = [
            s.strip()
            for s in data.get("sentences", [])
            if isinstance(s, str) and s.strip()
        ]

        if not sentences:
            summary_text = str(data.get("summary", "")).strip()
            sentences = split_sentences(summary_text)

        summary = " ".join(sentences).strip()
        return {"sentences": sentences, "summary": summary}

    def generate_rag_answer(
        self,
        retrieved: List[RetrievedChunk],
        question: str,
    ) -> Dict[str, Any]:
        """
        Generate a retrieval-conditioned answer using only retrieved chunks.

        Returns:
            {
                "sentences": [...],
                "answer": "..."
            }
        """
        client = _openai_client(self.cfg)
        context = _chunks_to_context(retrieved)

        instruction = (
            "You are a retrieval-grounded academic assistant. "
            "STRICT RULES:\n"
            "1) Answer ONLY using the provided chunks.\n"
            "2) Do NOT add outside knowledge.\n"
            "3) If the answer is not supported by the chunks, say so clearly.\n"
            "4) Write in British English.\n"
        )

        prompt = (
            f"{instruction}\n"
            f"Write approximately {self.cfg.ANSWER_SENTENCE_TARGET} sentences.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CHUNKS:\n{context}\n\n"
            "Return JSON with:\n"
            "- sentences: array of strings\n"
            "- answer: string\n"
        )

        schema = {
            "type": "object",
            "properties": {
                "sentences": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "answer": {"type": "string"},
            },
            "required": ["sentences", "answer"],
            "additionalProperties": False,
        }

        messages = [
            {"role": "system", "content": "Follow instructions exactly."},
            {"role": "user", "content": prompt},
        ]

        data = _try_structured_json(client, self.cfg, messages, "rag_answer", schema)

        sentences = [
            s.strip()
            for s in data.get("sentences", [])
            if isinstance(s, str) and s.strip()
        ]

        if not sentences:
            answer_text = str(data.get("answer", "")).strip()
            sentences = split_sentences(answer_text)

        answer = " ".join(sentences).strip()
        return {"sentences": sentences, "answer": answer}

    def generate_baseline_summary(self, full_text: str) -> str:
        """
        Generate a baseline summary with no retrieval grounding.

        Short text:
            summarise directly

        Long text:
            map-reduce style summarisation using BASELINE_MAP_CHUNK_CHARS
        """
        client = _openai_client(self.cfg)
        text = (full_text or "").strip()

        if not text:
            return ""

        if self.cfg.BASELINE_MODE.lower() == "textrank":
            # Placeholder fallback if TextRank is handled elsewhere in the app.
            return text[: min(len(text), 1200)]

        if len(text) <= self.cfg.BASELINE_MAX_CHARS:
            completion = client.chat.completions.create(
                model=self.cfg.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic summarisation assistant. "
                            "Summarise the provided text clearly in British English."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=self.cfg.OPENAI_TEMPERATURE,
                max_tokens=self.cfg.OPENAI_MAX_OUTPUT_TOKENS,
            )
            return (completion.choices[0].message.content or "").strip()

        partial_summaries: List[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.cfg.BASELINE_MAP_CHUNK_CHARS, len(text))
            chunk = text[start:end]

            completion = client.chat.completions.create(
                model=self.cfg.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic summarisation assistant. "
                            "Summarise the provided text chunk clearly in British English."
                        ),
                    },
                    {"role": "user", "content": chunk},
                ],
                temperature=self.cfg.OPENAI_TEMPERATURE,
                max_tokens=self.cfg.OPENAI_MAX_OUTPUT_TOKENS,
            )

            partial = (completion.choices[0].message.content or "").strip()
            if partial:
                partial_summaries.append(partial)

            start = end

        combined = "\n\n".join(partial_summaries).strip()
        if not combined:
            return ""

        completion = client.chat.completions.create(
            model=self.cfg.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an academic summarisation assistant. "
                        "Combine the partial summaries into one coherent overall summary "
                        "in British English."
                    ),
                },
                {"role": "user", "content": combined},
            ],
            temperature=self.cfg.OPENAI_TEMPERATURE,
            max_tokens=self.cfg.OPENAI_MAX_OUTPUT_TOKENS,
        )

        return (completion.choices[0].message.content or "").strip()