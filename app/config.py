from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_source_dir() -> Path:
    for candidate in (PROJECT_ROOT / "Data", PROJECT_ROOT / "data", PROJECT_ROOT / "data" / "chapters"):
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "Data"


CHAPTERS_DIR = _resolve_source_dir()
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "tourism_management"
EVAL_DATA_PATH = PROJECT_ROOT / "data" / "test_questions.csv"
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store.json"

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text:latest"

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 220
TOP_K = 4

