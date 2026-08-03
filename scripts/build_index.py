from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHAPTERS_DIR
from app.rag_pipeline import SejarahRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or rebuild the Sejarah RAG index.")
    parser.add_argument("--reset", action="store_true", help="Delete the current collection before indexing.")
    args = parser.parse_args()

    rag = SejarahRAG(CHAPTERS_DIR)
    count = rag.ingest(reset=args.reset)
    print(f"Indexed {count} chunks from {CHAPTERS_DIR}.")


if __name__ == "__main__":
    main()
