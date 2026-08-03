import os
import tempfile
import unittest
from pathlib import Path

from app.rag_pipeline import should_rebuild_index


class ShouldRebuildIndexTests(unittest.TestCase):
    def test_returns_true_when_index_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            index_path = root / "vector_store.json"
            source_path = root / "chapter.txt"
            source_path.write_text("Isi kandungan ujian", encoding="utf-8")

            self.assertTrue(should_rebuild_index(index_path, [source_path]))

    def test_returns_true_when_source_is_newer_than_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            index_path = root / "vector_store.json"
            source_path = root / "chapter.txt"
            source_path.write_text("Isi kandungan ujian", encoding="utf-8")
            index_path.write_text("{}", encoding="utf-8")

            future_time = 1_700_000_000
            os.utime(index_path, (future_time, future_time))
            os.utime(source_path, (future_time + 10, future_time + 10))

            self.assertTrue(should_rebuild_index(index_path, [source_path]))

    def test_returns_false_when_index_is_newer_than_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            index_path = root / "vector_store.json"
            source_path = root / "chapter.txt"
            source_path.write_text("Isi kandungan ujian", encoding="utf-8")
            index_path.write_text("{}", encoding="utf-8")

            future_time = 1_700_000_000
            os.utime(index_path, (future_time, future_time))
            os.utime(source_path, (future_time - 100, future_time - 100))

            self.assertFalse(should_rebuild_index(index_path, [source_path]))


if __name__ == "__main__":
    unittest.main()
