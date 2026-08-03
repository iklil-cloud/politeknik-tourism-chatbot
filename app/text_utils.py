from __future__ import annotations

import re
from typing import Iterable


_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SECTION_RE = re.compile(r"^(?P<section>\d+(?:\.\d+)+)\s+(?P<title>.+)")
_TAGGED_TITLE_RE = re.compile(r"^\[(?P<tag>TAJUK|SEKSYEN|SUBSEKSYEN)(?::\s*(?P<title>[^\]]+))?\]\s*(?P<rest>.*)$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    replacements = {
        "\ufeff": "",
        "\u00a0": " ",
        "â€¢": "-",
        "•": "-",
        "\r\n": "\n",
        "\r": "\n",
        "\t": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _MULTI_NEWLINE_RE.sub("\n\n", text).strip()


def iter_paragraphs(text: str) -> Iterable[str]:
    for paragraph in text.split("\n\n"):
        cleaned = paragraph.strip()
        if cleaned:
            yield cleaned


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def detect_chapter(paragraph: str) -> str | None:
    match = _TAGGED_TITLE_RE.match(paragraph.strip())
    if match and match.group("tag").upper() == "TAJUK":
        title = " ".join(part for part in [match.group("title"), match.group("rest")] if part).strip()
        if title:
            return title.strip(". ").strip()

    if paragraph.upper().startswith("BAB "):
        return paragraph.strip(". ").strip()
    return None


def detect_section(paragraph: str) -> tuple[str, str] | None:
    tagged_match = _TAGGED_TITLE_RE.match(paragraph.strip())
    if tagged_match and tagged_match.group("tag").upper() in {"SEKSYEN", "SUBSEKSYEN"}:
        title = " ".join(
            part for part in [tagged_match.group("title"), tagged_match.group("rest")] if part
        ).strip()
        if title:
            return title.strip(". "), title.strip(". ")

    match = _SECTION_RE.match(paragraph)
    if match:
        return match.group("section"), match.group("title").strip(". ")
    return None
