from __future__ import annotations

import json
import math
import re
from pathlib import Path

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class SimpleVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset(self) -> None:
        self.records = []
        self._save()

    def count(self) -> int:
        return len(self.records)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        for item_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            self.records.append(
                {
                    "id": item_id,
                    "document": document,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            )
        self._save()

    def query(self, query_embedding: list[float], query_text: str, n_results: int) -> dict[str, list[list[object]]]:
        scored: list[tuple[float, dict[str, object]]] = []
        query_tokens = set(tokenize_text(query_text))
        for record in self.records:
            embedding_score = cosine_similarity(query_embedding, list(record["embedding"]))
            keyword_score = token_overlap_score(query_tokens, tokenize_text(str(record["document"])))
            phrase_score = phrase_match_score(query_text, str(record["document"]))
            score = (0.6 * embedding_score) + (0.2 * keyword_score) + (0.2 * phrase_score)
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)

        top_records = scored[:n_results]
        return {
            "documents": [[str(item[1]["document"]) for item in top_records]],
            "metadatas": [[dict(item[1]["metadata"]) for item in top_records]],
            "distances": [[1 - item[0] for item in top_records]],
        }

    def get_by_chapter(self, chapter: str) -> dict[str, list[object]]:
        matches = [
            record for record in self.records if str(record["metadata"].get("chapter", "")) == chapter
        ]
        return {
            "documents": [str(record["document"]) for record in matches],
            "metadatas": [dict(record["metadata"]) for record in matches],
        }

    def list_chapters(self) -> list[str]:
        chapters = {
            str(record["metadata"].get("chapter", ""))
            for record in self.records
            if record.get("metadata")
        }
        return sorted(chapter for chapter in chapters if chapter)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_overlap_score(query_tokens: set[str], document_tokens: list[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    document_token_set = set(document_tokens)
    overlap = len(query_tokens & document_token_set)
    return overlap / len(query_tokens)


def phrase_match_score(query_text: str, document_text: str) -> float:
    query_lower = query_text.lower().strip(" ?.")
    document_lower = document_text.lower()

    if query_lower in document_lower:
        return 1.0

    if "maksud " in query_lower:
        term = query_lower.split("maksud ", 1)[1].strip(" ?.")
        if term:
            if f"{term} bermaksud" in document_lower:
                return 1.0
            if f"{term} ialah" in document_lower:
                return 0.9
            if "bermaksud" in document_lower and term in document_lower:
                return 0.8

    return 0.0
