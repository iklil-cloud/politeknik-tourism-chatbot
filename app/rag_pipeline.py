from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_MODEL,
    TOP_K,
    VECTOR_STORE_PATH,
)
from app.data_loader import DocumentChunk, build_chunks, iter_source_files
from app.ollama_client import OllamaClient
from app.prompts import build_qa_prompt, build_summary_prompt
from app.simple_store import SimpleVectorStore

try:
    import chromadb
except Exception:
    chromadb = None


@dataclass
class RetrievalResult:
    text: str
    chapter: str
    section: str
    title: str
    source: str
    distance: float | None


def should_rebuild_index(index_path: Path, source_files: list[Path]) -> bool:
    if not index_path.exists() or index_path.stat().st_size == 0:
        return True

    index_time = index_path.stat().st_mtime
    for source_file in source_files:
        if not source_file.exists():
            continue
        source_time = source_file.stat().st_mtime
        if source_time > index_time + 1e-6:
            return True
    return False


class SejarahRAG:
    def __init__(
        self,
        source_path: Path,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        llm_model: str = LLM_MODEL,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        self.source_path = source_path
        self.ollama = OllamaClient()
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.backend = "simple"
        self.store = SimpleVectorStore(VECTOR_STORE_PATH)
        self.client = None
        self.collection = None

        if chromadb is not None:
            try:
                chroma_dir.mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=str(chroma_dir))
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Sejarah Form 5 textbook chunks"},
                )
                self.backend = "chroma"
            except Exception:
                self.client = None
                self.collection = None

    def ingest(self, reset: bool = False) -> int:
        if self.backend == "chroma":
            return self._ingest_chroma(reset)
        return self._ingest_simple(reset)

    def _ingest_chroma(self, reset: bool = False) -> int:
        if reset and self.client is not None and self.collection is not None:
            try:
                self.client.delete_collection(self.collection.name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name=self.collection.name)

        if self.collection is not None and self.collection.count() > 0 and not reset:
            return self.collection.count()

        chunks = build_chunks(self.source_path)
        for chunk in chunks:
            embedding = self.ollama.embed(self.embed_model, chunk.text)
            self.collection.add(
                ids=[chunk.chunk_id],
                documents=[chunk.text],
                embeddings=[embedding],
                metadatas=[chunk_to_metadata(chunk)],
            )
        return len(chunks)

    def _ingest_simple(self, reset: bool = False) -> int:
        if reset:
            self.store.reset()

        source_files = iter_source_files(self.source_path)
        if self.store.count() > 0 and not reset and not should_rebuild_index(self.store.path, source_files):
            return self.store.count()

        if self.store.count() > 0 and not reset:
            self.store.reset()

        chunks = build_chunks(self.source_path)
        for chunk in chunks:
            embedding = self.ollama.embed(self.embed_model, chunk.text)
            self.store.add(
                ids=[chunk.chunk_id],
                documents=[chunk.text],
                embeddings=[embedding],
                metadatas=[chunk_to_metadata(chunk)],
            )
        return len(chunks)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[RetrievalResult]:
        query_embedding = self.ollama.embed(self.embed_model, question)
        if self.backend == "chroma":
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
        else:
            results = self.store.query(
                query_embedding=query_embedding,
                query_text=question,
                n_results=top_k,
            )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievalResult] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            retrieved.append(
                RetrievalResult(
                    text=document,
                    chapter=str(metadata.get("chapter", "")),
                    section=str(metadata.get("section", "")),
                    title=str(metadata.get("title", "")),
                    source=str(metadata.get("source", "")),
                    distance=distance,
                )
            )
        return retrieved

    def answer_question(self, question: str, top_k: int = TOP_K) -> dict[str, object]:
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return {
                "answer": "Maaf, saya tidak menemui maklumat yang relevan dalam data rujukan.",
                "sources": [],
            }

        context = build_context(retrieved)
        prompt = build_qa_prompt(question, context)
        answer = self.ollama.generate(
            self.llm_model,
            prompt,
            temperature=0.2,
            num_predict=350,
        )
        if not answer.strip():
            answer = self._fallback_answer(question, retrieved)
        return {"answer": answer, "sources": retrieved}

    def summarize_chapter(self, chapter: str) -> dict[str, object]:
        if self.backend == "chroma":
            results = self.collection.get(
                where={"chapter": chapter},
                include=["documents", "metadatas"],
            )
        else:
            results = self.store.get_by_chapter(chapter)
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        if not documents:
            return {
                "summary": "Tiada kandungan ditemui untuk bab yang dipilih.",
                "sources": [],
            }

        retrieved = [
            RetrievalResult(
                text=document,
                chapter=str(metadata.get("chapter", "")),
                section=str(metadata.get("section", "")),
                title=str(metadata.get("title", "")),
                source=str(metadata.get("source", "")),
                distance=None,
            )
            for document, metadata in zip(documents, metadatas)
        ]
        summary_sources = retrieved[:10]
        context = build_context(summary_sources, max_chars=1600)
        prompt = build_summary_prompt(chapter, context)
        summary = self.ollama.generate(
            self.llm_model,
            prompt,
            temperature=0.2,
            num_predict=900,
        )
        if not summary.strip():
            summary = self._fallback_summary(chapter, summary_sources)
        return {"summary": summary, "sources": summary_sources}

    def list_chapters(self) -> list[str]:
        if self.backend == "chroma":
            results = self.collection.get(include=["metadatas"])
            chapters = {str(item.get("chapter", "")) for item in results.get("metadatas", [])}
            return sorted(chapter for chapter in chapters if chapter)
        return self.store.list_chapters()

    def _fallback_answer(self, question: str, retrieved: list[RetrievalResult]) -> str:
        if not retrieved:
            return "Maaf, saya tidak menemui maklumat yang relevan dalam data rujukan."

        top_text = re.sub(r"\s+", " ", retrieved[0].text).strip()
        if not top_text:
            return "Saya tidak menemui kandungan yang sesuai untuk soalan ini."

        first_sentence = re.split(r"(?<=[.!?])\s+", top_text)[0]
        if len(first_sentence) > 220:
            first_sentence = first_sentence[:217] + "..."
        return f"Berdasarkan kandungan rujukan, {first_sentence}"

    def _fallback_summary(self, chapter: str, retrieved: list[RetrievalResult]) -> str:
        if not retrieved:
            return f"Tiada kandungan ditemui untuk {chapter}."

        parts = [f"{chapter}:"]
        for item in retrieved[:3]:
            text = re.sub(r"\s+", " ", item.text).strip()
            if text:
                parts.append(f"- {text[:180]}{'...' if len(text) > 180 else ''}")
        return "\n".join(parts)


def chunk_to_metadata(chunk: DocumentChunk) -> dict[str, str]:
    return {
        "chapter": chunk.chapter,
        "section": chunk.section,
        "title": chunk.title,
        "source": chunk.source,
    }


def build_context(results: list[RetrievalResult], max_chars: int = 900) -> str:
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        heading = item.source
        if item.title:
            heading = f"{heading} - {item.title}"
        snippet = item.text[:max_chars].strip()
        lines.append(f"[Sumber {index}] {heading}\n{snippet}")
    return "\n\n".join(lines)
