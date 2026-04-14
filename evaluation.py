from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

from chunking import chunk_units
from config import AppConfig, get_config
from generator import LLMGenerator
from ingestion import ingest_files
from models import DocumentUnit
from retrieval import VectorIndex
from text_utils import count_words, safe_join_text
from traceability import NLIJudge, attribute_sentences, hallucination_rate, mean_grounding
from visualisation import plot_correlation_heatmap, plot_scatter, save_fig


def _load_csv_dataset(cfg: AppConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load local news_summary.csv if present.
    Expected columns (common variant): ctext=article, text=summary.
    """
    df = pd.read_csv(cfg.NEWS_CSV_PATH, encoding="latin-1")
    # common column names from earlier pipeline
    if "ctext" in df.columns and "text" in df.columns:
        df = df.rename(columns={"ctext": "document", "text": "summary"})
    elif cfg.HF_TEXT_FIELD in df.columns and cfg.HF_SUMMARY_FIELD in df.columns:
        df = df.rename(columns={cfg.HF_TEXT_FIELD: "document", cfg.HF_SUMMARY_FIELD: "summary"})
    else:
        raise ValueError("CSV dataset missing expected columns for document/summary.")

    df = df.dropna(subset=["document", "summary"]).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=cfg.RANDOM_SEED).reset_index(drop=True)

    split = int(len(df) * cfg.TRAIN_TEST_RATIO)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    train = [{"id": f"train_{i}", "document": r["document"], "summary": r["summary"]} for i, r in train_df.iterrows()]
    test = [{"id": f"test_{i}", "document": r["document"], "summary": r["summary"]} for i, r in test_df.iterrows()]

    train = train[: cfg.SAMPLE_TRAIN]
    test = test[: cfg.SAMPLE_TEST]
    return train, test


def _load_hf_dataset(cfg: AppConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load a public summarisation dataset via Hugging Face datasets.
    Default: xsum (document/summary).
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.HF_DATASET_NAME)

    if "train" in ds and "test" in ds:
        train_ds = ds["train"]
        test_ds = ds["test"]
    else:
        # fallback split
        train_test = ds[list(ds.keys())[0]].train_test_split(test_size=1.0 - cfg.TRAIN_TEST_RATIO, seed=cfg.RANDOM_SEED)
        train_ds = train_test["train"]
        test_ds = train_test["test"]

    def to_rows(split_ds, prefix: str, limit: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for i, ex in enumerate(split_ds):
            doc = str(ex.get(cfg.HF_TEXT_FIELD, "")).strip()
            summ = str(ex.get(cfg.HF_SUMMARY_FIELD, "")).strip()
            if doc and summ:
                rows.append({"id": f"{prefix}_{i}", "document": doc, "summary": summ})
            if len(rows) >= limit:
                break
        return rows

    train = to_rows(train_ds, "train", cfg.SAMPLE_TRAIN)
    test = to_rows(test_ds, "test", cfg.SAMPLE_TEST)
    return train, test


def prepare_dataset(cfg: AppConfig | None = None) -> Tuple[Path, Path]:
    """
    Prepare processed train/test JSON files under data/processed/.
    """
    cfg = cfg or get_config()
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mode = cfg.DATASET_MODE.lower().strip()
    if mode == "csv" or (mode == "auto" and cfg.NEWS_CSV_PATH.exists()):
        train, test = _load_csv_dataset(cfg)
    else:
        train, test = _load_hf_dataset(cfg)

    train_path = cfg.PROCESSED_DIR / "train.json"
    test_path = cfg.PROCESSED_DIR / "test.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    return train_path, test_path


def _rouge(pred: str, ref: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {
        "rouge1_f": float(scores["rouge1"].fmeasure),
        "rouge2_f": float(scores["rouge2"].fmeasure),
        "rougeL_f": float(scores["rougeL"].fmeasure),
    }


def _semantic_similarity(embed_fn, a: str, b: str) -> float:
    import numpy as np
    ea = embed_fn([a])[0]
    eb = embed_fn([b])[0]
    return float(np.dot(ea, eb))


def _build_index_for_document(doc_text: str, cfg: AppConfig) -> Tuple[VectorIndex, List[DocumentUnit]]:
    units = [
        DocumentUnit(
            unit_id=0,
            text=doc_text,
            source="dataset_document",
            file_type="dataset",
            location_type="section",
            location="whole document",
            location_index=None,
        )
    ]
    chunks = chunk_units(units, cfg=cfg)
    index = VectorIndex.create(cfg=cfg)
    index.build(chunks)
    return index, units


def run_dataset_evaluation(
    cfg: AppConfig | None = None,
    limit: int = 25,
) -> pd.DataFrame:
    """
    Evaluate baseline vs RAG on processed test set.
    """
    cfg = cfg or get_config()
    test_path = cfg.PROCESSED_DIR / "test.json"
    if not test_path.exists():
        prepare_dataset(cfg)

    with open(test_path, encoding="utf-8") as f:
        test = json.load(f)

    generator = LLMGenerator.create(cfg)
    nli = NLIJudge(cfg.NLI_MODEL, enabled=cfg.ENABLE_NLI)

    rows: List[Dict[str, Any]] = []

    for ex in tqdm(test[:limit], desc="Evaluating", unit="doc"):
        doc = ex["document"]
        ref = ex["summary"]

        index, _ = _build_index_for_document(doc, cfg)
        retrieved = index.topk_for_summary_query()

        rag_out = generator.generate_rag_summary(retrieved=retrieved)
        rag_sentences = rag_out["sentences"]
        rag_summary = rag_out["summary"]

        baseline_summary = generator.generate_baseline_summary(doc)

        # Traceability (for hallucination proxy + grounding stats)
        attributions = attribute_sentences(
            sentences=rag_sentences,
            retrieved=retrieved,
            embed_fn=index.embed,
            cfg=cfg,
            nli=nli,
        )

        # Evidence size ratio proxy
        evidence_words = sum(count_words(r.chunk.text) for r in retrieved)
        doc_words = count_words(doc)
        evidence_ratio = float(evidence_words / doc_words) if doc_words else 0.0

        # Reading-time proxy (seconds)
        wpm = max(60, cfg.WORDS_PER_MINUTE)
        est_doc_time = (doc_words / wpm) * 60.0
        est_ev_time = (evidence_words / wpm) * 60.0

        r_rouge = _rouge(rag_summary, ref)
        b_rouge = _rouge(baseline_summary, ref)

        row = {
            "id": ex["id"],
            "doc_words": doc_words,
            "evidence_words": evidence_words,
            "evidence_ratio": evidence_ratio,
            "est_doc_time_s": est_doc_time,
            "est_evidence_time_s": est_ev_time,
            "rag_sem_sim": _semantic_similarity(index.embed, rag_summary, ref),
            "baseline_sem_sim": _semantic_similarity(index.embed, baseline_summary, ref),
            "rag_grounding_mean": mean_grounding(attributions),
            "rag_hallucination_rate": hallucination_rate(attributions),
        }
        row.update({f"rag_{k}": v for k, v in r_rouge.items()})
        row.update({f"baseline_{k}": v for k, v in b_rouge.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def cli() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Capstone evaluation CLI")
    parser.add_argument("--prepare_dataset", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out_csv", type=str, default=str(cfg.PROCESSED_DIR / "eval_results.csv"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--in_csv", type=str, default=str(cfg.PROCESSED_DIR / "eval_results.csv"))
    parser.add_argument("--out_dir", type=str, default=str(cfg.PROCESSED_DIR / "plots"))
    args = parser.parse_args()

    if args.prepare_dataset:
        train_path, test_path = prepare_dataset(cfg)
        print(f"Saved: {train_path}")
        print(f"Saved: {test_path}")

    if args.run_eval:
        df = run_dataset_evaluation(cfg, limit=args.limit)
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved results: {out}")

    if args.plot:
        df = pd.read_csv(args.in_csv)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig_h = plot_correlation_heatmap(df, title="Evaluation Metrics Correlation")
        save_fig(fig_h, out_dir / "corr_heatmap.png")

        if "rag_hallucination_rate" in df.columns and "rag_grounding_mean" in df.columns:
            fig_s = plot_scatter(df, x="rag_grounding_mean", y="rag_hallucination_rate",
                                 title="Hallucination vs Grounding")
            save_fig(fig_s, out_dir / "hallucination_vs_grounding.png")

        print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    cli()