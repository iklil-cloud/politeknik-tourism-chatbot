from __future__ import annotations

from pathlib import Path

from app.data_loader import iter_source_files


def test_iter_source_files_includes_pdf_and_txt(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("isi kandungan teks", encoding="utf-8")
    (tmp_path / "chapter.pdf").write_bytes(b"%PDF-1.4\n%test")

    files = iter_source_files(tmp_path)

    assert [path.name for path in files] == ["chapter.pdf", "notes.txt"]
