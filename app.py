from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from chunking import chunk_units
from config import get_config
from evaluation import prepare_dataset, run_dataset_evaluation
from generator import LLMGenerator
from ingestion import ingest_uploaded_files
from logging_utils import append_jsonl
from models import RetrievedChunk, SentenceAttribution, VerificationEvent
from retrieval import VectorIndex
from study_tools import (
    generate_checklist,
    generate_flashcards,
    generate_study_plan,
    generate_timeline,
)
from text_utils import count_words, safe_join_text, truncate_chars
from traceability import NLIJudge, attribute_sentences, hallucination_rate, mean_grounding
from visualisation import plot_correlation_heatmap, plot_scatter


cfg = get_config()
cfg.ensure_dirs()


st.set_page_config(page_title="Trustworthy Traceable RAG Study Tool", layout="wide")


@st.cache_resource
def get_generator() -> LLMGenerator:
    return LLMGenerator.create(cfg)


@st.cache_resource
def get_blank_index() -> VectorIndex:
    return VectorIndex.create(cfg)


def _log_event(ev: VerificationEvent) -> None:
    append_jsonl(cfg.LOGS_DIR / "verification_events.jsonl", ev.to_dict())


def _ensure_session() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def _build_index_from_uploads(uploaded_files) -> Dict[str, Any]:
    units = ingest_uploaded_files(uploaded_files, cfg=cfg)
    chunks = chunk_units(units, cfg=cfg)
    index = VectorIndex.create(cfg)
    index.build(chunks)
    return {"units": units, "chunks": chunks, "index": index}


def _combine_units_text(units) -> str:
    return safe_join_text([u.text for u in units], sep="\n\n")


def _render_attributed_output(
    sentences: List[str],
    retrieved: List[RetrievedChunk],
    index: VectorIndex,
    title: str,
    start_time: float,
    source_label: str,
) -> List[SentenceAttribution]:
    st.subheader(title)

    nli = NLIJudge(cfg.NLI_MODEL, enabled=cfg.ENABLE_NLI)
    attributions = attribute_sentences(
        sentences=sentences,
        retrieved=retrieved,
        embed_fn=index.embed,
        cfg=cfg,
        nli=nli,
        top_m=3,
    )

    st.markdown(
        f"**Mean grounding (cosine):** `{mean_grounding(attributions):.3f}`  \n"
        f"**Hallucination proxy rate:** `{hallucination_rate(attributions):.2%}`"
    )

    for a in attributions:
        with st.expander(f"Sentence {a.sentence_id + 1} | supported={a.supported} | score={a.best_similarity:.3f}"):
            st.write(a.sentence)
            st.caption(f"Citation: {a.citation or '—'}")
            if a.entailment_label is not None:
                st.caption(f"NLI: {a.entailment_label} ({a.entailment_prob:.2f})")

            st.markdown("**Evidence snippet**")
            st.write(a.evidence_snippet or "—")

            # Verification-effort (time-to-verify)
            cols = st.columns(3)
            elapsed = time.time() - start_time
            session_id = _ensure_session()

            with cols[0]:
                if st.button("Mark verified", key=f"verify_{source_label}_{a.sentence_id}"):
                    ev = VerificationEvent(
                        session_id=session_id,
                        event_type="verified",
                        timestamp_unix=time.time(),
                        elapsed_seconds=elapsed,
                        source=source_label,
                        sentence_id=a.sentence_id,
                        sentence=a.sentence,
                        chunk_id=a.best_chunk_id,
                        citation=a.citation,
                    )
                    _log_event(ev)
                    st.success(f"Logged verified at {elapsed:.1f}s")

            with cols[1]:
                if st.button("Flag unsupported", key=f"flag_{source_label}_{a.sentence_id}"):
                    ev = VerificationEvent(
                        session_id=session_id,
                        event_type="flagged",
                        timestamp_unix=time.time(),
                        elapsed_seconds=elapsed,
                        source=source_label,
                        sentence_id=a.sentence_id,
                        sentence=a.sentence,
                        chunk_id=a.best_chunk_id,
                        citation=a.citation,
                    )
                    _log_event(ev)
                    st.warning(f"Logged flagged at {elapsed:.1f}s")

            with cols[2]:
                if st.button("Opened evidence", key=f"open_{source_label}_{a.sentence_id}"):
                    ev = VerificationEvent(
                        session_id=session_id,
                        event_type="opened_evidence",
                        timestamp_unix=time.time(),
                        elapsed_seconds=elapsed,
                        source=source_label,
                        sentence_id=a.sentence_id,
                        sentence=a.sentence,
                        chunk_id=a.best_chunk_id,
                        citation=a.citation,
                    )
                    _log_event(ev)
                    st.info(f"Logged evidence open at {elapsed:.1f}s")

            if a.candidates:
                st.markdown("**Top evidence candidates**")
                st.dataframe(pd.DataFrame(a.candidates), use_container_width=True)

    return attributions


def page_documents():
    st.title("Trustworthy Traceable RAG Academic Summaries")

    st.sidebar.header("Index settings")
    st.sidebar.write(f"Chunk size (words): {cfg.CHUNK_SIZE_WORDS}")
    st.sidebar.write(f"Overlap (words): {cfg.CHUNK_OVERLAP_WORDS}")
    st.sidebar.write(f"Top-k retrieval: {cfg.TOP_K_RETRIEVAL}")
    st.sidebar.write(f"Embedding model: {cfg.EMBEDDING_MODEL}")

    uploaded = st.file_uploader(
        "Upload academic materials (PDF, DOCX, PPTX, TXT, MD, CSV)",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "txt", "md", "csv"],
        help="Multiple uploads supported; provenance is preserved (page/paragraph/slide/row).",
    )
    if not uploaded:
        st.info("Upload one or more files to build an index.")
        return

    if st.button("Build retrieval index"):
        with st.spinner("Ingesting, chunking, embedding, and building FAISS index..."):
            bundle = _build_index_from_uploads(uploaded)
        st.session_state["bundle"] = bundle
        st.success("Index built.")

    bundle = st.session_state.get("bundle")
    if not bundle:
        st.warning("Build the index first.")
        return

    units = bundle["units"]
    chunks = bundle["chunks"]
    index: VectorIndex = bundle["index"]

    st.markdown(
        f"**Sources:** `{len(set(u.source for u in units))}`  \n"
        f"**Provenance units:** `{len(units)}`  \n"
        f"**Chunks:** `{len(chunks)}`"
    )

    generator = get_generator()
    full_text = _combine_units_text(units)

    tab1, tab2, tab3 = st.tabs(["Summaries", "Ask a question", "Study tools"])

    with tab1:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Baseline summary (no retrieval)")
            if st.button("Generate baseline summary"):
                with st.spinner("Generating baseline summary..."):
                    bsum = generator.generate_baseline_summary(full_text)
                st.session_state["baseline_summary"] = bsum

            if "baseline_summary" in st.session_state:
                st.write(st.session_state["baseline_summary"])

        with colB:
            st.subheader("RAG summary (retrieval-conditioned)")
            focus = st.text_input(
                "Optional focus for summary (leave blank for general summary)",
                value="",
                help="Used as retrieval query and summarisation focus in RAG mode.",
            )

            if st.button("Generate RAG summary"):
                with st.spinner("Retrieving top-k chunks..."):
                    retrieved = index.topk_for_summary_query(focus=focus or None)
                with st.spinner("Generating RAG summary using ONLY retrieved chunks..."):
                    rag = generator.generate_rag_summary(retrieved=retrieved, focus=focus or None)

                st.session_state["rag_retrieved"] = retrieved
                st.session_state["rag_output"] = rag
                st.session_state["rag_start_time"] = time.time()

            if "rag_output" in st.session_state and "rag_retrieved" in st.session_state:
                rag = st.session_state["rag_output"]
                retrieved = st.session_state["rag_retrieved"]
                st.write(rag["summary"])

                st.markdown("**Retrieved evidence used**")
                st.dataframe(
                    pd.DataFrame([r.to_dict() for r in retrieved]),
                    use_container_width=True,
                )

                start_time = st.session_state.get("rag_start_time", time.time())
                _render_attributed_output(
                    sentences=rag["sentences"],
                    retrieved=retrieved,
                    index=index,
                    title="Sentence-level citations and grounding (RAG output)",
                    start_time=start_time,
                    source_label="rag_summary",
                )

    with tab2:
        st.subheader("Retrieval-conditioned Q&A")
        question = st.text_input("Ask a question about your uploaded documents:")
        if st.button("Answer with RAG") and question.strip():
            with st.spinner("Retrieving evidence..."):
                retrieved = index.search(question, top_k=cfg.TOP_K_RETRIEVAL)
            with st.spinner("Generating answer using ONLY retrieved chunks..."):
                out = generator.generate_rag_answer(retrieved=retrieved, question=question)

            st.session_state["qa_retrieved"] = retrieved
            st.session_state["qa_output"] = out
            st.session_state["qa_start_time"] = time.time()

        if "qa_output" in st.session_state and "qa_retrieved" in st.session_state:
            out = st.session_state["qa_output"]
            retrieved = st.session_state["qa_retrieved"]
            st.write(out["answer"])

            start_time = st.session_state.get("qa_start_time", time.time())
            _render_attributed_output(
                sentences=out["sentences"],
                retrieved=retrieved,
                index=index,
                title="Sentence-level citations and grounding (RAG answer)",
                start_time=start_time,
                source_label="rag_qa",
            )

    with tab3:
        st.subheader("Study artefacts from retrieved evidence")

        if "rag_retrieved" not in st.session_state:
            st.info("Generate a RAG summary first to derive study tools from retrieved evidence.")
            return

        evidence_text = "\n".join([r.chunk.text for r in st.session_state["rag_retrieved"]])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Flashcards")
            cards = generate_flashcards(evidence_text)
            for q, a in cards:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")

            st.markdown("### Checklist")
            for item in generate_checklist(evidence_text):
                st.markdown(f"- {item}")

        with col2:
            st.markdown("### Timeline")
            timeline = generate_timeline(evidence_text)
            if timeline:
                for ev in timeline:
                    st.markdown(f"- {ev}")
            else:
                st.write("No date-like events found.")

            st.markdown("### Study plan")
            days = st.slider("Days", min_value=1, max_value=14, value=5)
            sessions = st.slider("Sessions per day", min_value=1, max_value=3, value=2)
            plan = generate_study_plan(evidence_text, days=days, sessions_per_day=sessions)
            st.dataframe(pd.DataFrame(plan), use_container_width=True)


def page_dataset_eval():
    st.title("Dataset Evaluation (Baseline vs RAG)")

    st.write(
        "This page downloads/uses a summarisation dataset, prepares train/test JSON files, "
        "runs baseline vs RAG evaluation on a subset, and produces plots."
    )

    colA, colB = st.columns(2)

    with colA:
        if st.button("Prepare dataset (train/test JSON)"):
            with st.spinner("Preparing dataset..."):
                train_path, test_path = prepare_dataset(cfg)
            st.success(f"Prepared:\n- {train_path}\n- {test_path}")

    with colB:
        limit = st.slider("Evaluation limit (documents)", min_value=5, max_value=50, value=15)
        if st.button("Run evaluation"):
            with st.spinner("Running evaluation (this will call the LLM)..."):
                df = run_dataset_evaluation(cfg, limit=limit)
            st.session_state["eval_df"] = df
            st.success("Evaluation complete.")

    if "eval_df" not in st.session_state:
        st.info("Run evaluation to view results and plots.")
        return

    df: pd.DataFrame = st.session_state["eval_df"]
    st.subheader("Per-document metrics")
    st.dataframe(df, use_container_width=True)

    st.subheader("Correlation heatmap")
    fig = plot_correlation_heatmap(df, title="Correlation of evaluation metrics")
    st.pyplot(fig)

    if "rag_grounding_mean" in df.columns and "rag_hallucination_rate" in df.columns:
        st.subheader("Scatter: hallucination vs grounding")
        fig2 = plot_scatter(df, x="rag_grounding_mean", y="rag_hallucination_rate")
        st.pyplot(fig2)


def page_logs():
    st.title("Verification logs (JSONL)")
    log_path = cfg.LOGS_DIR / "verification_events.jsonl"
    if not log_path.exists():
        st.info("No logs yet. Verify or flag sentences in the Documents page first.")
        return
    lines = log_path.read_text(encoding="utf-8").splitlines()
    st.caption(str(log_path))
    st.text_area("Logs", "\n".join(lines[-300:]), height=420)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Documents", "Dataset evaluation", "Logs"],
    )

    if page == "Documents":
        page_documents()
    elif page == "Dataset evaluation":
        page_dataset_eval()
    else:
        page_logs()


if __name__ == "__main__":
    main()