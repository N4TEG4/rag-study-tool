from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import streamlit as st
except Exception:
    st = None

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    """
    Central configuration for the capstone prototype.

    Design mapping to report requirements:
    - Chunking/top_k/thresholds are explicit and tunable for evaluation.
    - LLM provider is configurable (OpenAI default; local via OpenAI-compatible base_url).
    """
    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Ingestion limits (reasonable degree; configurable)
    MAX_FILE_MB: int = int(os.getenv("MAX_FILE_MB", "30"))
    MAX_TOTAL_CHUNKS: int = int(os.getenv("MAX_TOTAL_CHUNKS", "6000"))

    # Chunking
    CHUNK_SIZE_WORDS: int = int(os.getenv("CHUNK_SIZE_WORDS", "180"))
    CHUNK_OVERLAP_WORDS: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))

    # Retrieval
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "6"))

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # LLM provider
    # openai = OpenAI hosted models
    # openai_compatible = local endpoint speaking OpenAI API 
    # huggingface = local transformers summarisation pipeline
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str | None = (
    st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if st is not None
    else os.getenv("OPENAI_API_KEY")
)
    OPENAI_BASE_URL: str | None = None
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    OPENAI_MAX_OUTPUT_TOKENS: int = 900

    # Prefer Structured Outputs (JSON Schema) when model supports it. Fall back otherwise
    OPENAI_USE_STRUCTURED_OUTPUTS: bool = _env_bool("OPENAI_STRUCTURED", True)

    # Generation targets
    SUMMARY_SENTENCE_TARGET: int = int(os.getenv("SUMMARY_SENTENCE_TARGET", "8"))
    ANSWER_SENTENCE_TARGET: int = int(os.getenv("ANSWER_SENTENCE_TARGET", "6"))

    # Grounding / hallucination proxy thresholds
    SUPPORT_SIM_THRESHOLD: float = float(os.getenv("SUPPORT_SIM_THRESHOLD", "0.28"))

    ENABLE_NLI: bool = _env_bool("ENABLE_NLI", False)
    NLI_MODEL: str = os.getenv("NLI_MODEL", "roberta-large-mnli")
    ENTAILMENT_THRESHOLD: float = float(os.getenv("ENTAILMENT_THRESHOLD", "0.55"))

    # Baselines
    BASELINE_MODE: str = os.getenv("BASELINE_MODE", "llm")  # llm|textrank
    BASELINE_MAX_CHARS: int = int(os.getenv("BASELINE_MAX_CHARS", "14000"))
    BASELINE_MAP_CHUNK_CHARS: int = int(os.getenv("BASELINE_MAP_CHUNK_CHARS", "5000"))

    # Dataset settings (evaluation)
    DATASET_MODE: str = os.getenv("DATASET_MODE", "auto")  # auto|csv|huggingface
    NEWS_CSV_PATH: Path = DATA_DIR / "news_summary.csv"
    HF_DATASET_NAME: str = os.getenv("HF_DATASET_NAME", "xsum")
    HF_TEXT_FIELD: str = os.getenv("HF_TEXT_FIELD", "document")
    HF_SUMMARY_FIELD: str = os.getenv("HF_SUMMARY_FIELD", "summary")
    TRAIN_TEST_RATIO: float = float(os.getenv("TRAIN_TEST_RATIO", "0.8"))
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    SAMPLE_TRAIN: int = int(os.getenv("SAMPLE_TRAIN", "500"))
    SAMPLE_TEST: int = int(os.getenv("SAMPLE_TEST", "100"))

    # Verification effort proxy
    WORDS_PER_MINUTE: int = int(os.getenv("WORDS_PER_MINUTE", "200"))

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


_CONFIG: AppConfig | None = None


def get_config() -> AppConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = AppConfig()
        _CONFIG.ensure_dirs()
    return _CONFIG