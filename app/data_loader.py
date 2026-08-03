from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency fallback
    PdfReader = None

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.text_utils import (
    detect_chapter,
    detect_section,
    iter_paragraphs,
    normalize_text,
    split_sentences,
)


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    chapter: str
    section: str
    title: str
    source: str


def load_source_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if PdfReader is None:
            return ""
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return normalize_text("\n".join(pages))

    if suffix == ".docx":
        try:
            with ZipFile(path) as archive:
                xml_bytes = archive.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for paragraph in root.findall(".//w:p", ns):
                texts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
                content = " ".join(texts).strip()
                if content:
                    paragraphs.append(content)
            return normalize_text("\n\n".join(paragraphs))
        except Exception:
            return ""

    return normalize_text(path.read_text(encoding="utf-8"))


def build_chunks(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0
    for source_file in iter_source_files(path):
        file_chunks = build_chunks_from_text(load_source_text(source_file), chunk_index)
        chunks.extend(file_chunks)
        chunk_index += len(file_chunks)
    return chunks


def iter_source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        supported = {".txt", ".pdf", ".docx"}
        files = sorted(
            item for item in path.iterdir() if item.is_file() and item.suffix.lower() in supported
        )
        if files:
            return files

    raise FileNotFoundError(f"No supported sources found at {path}")


def build_chunks_from_text(text: str, start_index: int = 0) -> list[DocumentChunk]:
    paragraphs = list(iter_paragraphs(text))

    chunks: list[DocumentChunk] = []
    current_chapter = "Bab tidak diketahui"
    current_section = ""
    current_title = ""
    buffer: list[str] = []
    chunk_index = start_index

    def flush_buffer() -> None:
        nonlocal buffer, chunk_index
        if not buffer:
            return

        combined = " ".join(buffer).strip()
        if not combined:
            buffer = []
            return

        for piece in split_into_windows(combined, CHUNK_SIZE, CHUNK_OVERLAP):
            source = current_section or current_chapter
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk-{chunk_index:03d}",
                    text=piece,
                    chapter=current_chapter,
                    section=current_section,
                    title=current_title,
                    source=source,
                )
            )
            chunk_index += 1
        buffer = []

    for paragraph in paragraphs:
        chapter = detect_chapter(paragraph)
        section = detect_section(paragraph.splitlines()[0].strip())

        if chapter:
            flush_buffer()
            current_chapter = chapter
            current_section = ""
            current_title = ""
            continue

        if section:
            flush_buffer()
            current_section, current_title = section
            lines = paragraph.splitlines()
            remaining = " ".join(line.strip() for line in lines[1:] if line.strip()).strip()
            if remaining:
                buffer.append(remaining)
            continue

        candidate = " ".join(buffer + [paragraph]).strip()
        if len(candidate) > CHUNK_SIZE and buffer:
            flush_buffer()

        buffer.append(paragraph)

    flush_buffer()
    return chunks


def split_into_windows(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    sentences = split_sentences(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > chunk_size:
            chunks.append(current.strip())
            overlap_text = current[-chunk_overlap:].strip()
            current = f"{overlap_text} {sentence}".strip()
        else:
            current = candidate

    if current:
        chunks.append(current.strip())

    return chunks
