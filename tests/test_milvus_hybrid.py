import json
import tempfile
import unittest
from pathlib import Path

from backend.milvus_hybrid import (
    build_path_filter,
    collect_chunk_documents,
    content_hash,
    document_id_for_path,
    parse_milvus_hits,
    source_path_for_chunk,
    summary_for_chunk,
)
from backend.embedding_provider import EmbeddingProvider


class MilvusHybridHelpersTest(unittest.TestCase):
    def test_local_hash_embedding_is_deterministic(self):
        provider = EmbeddingProvider(provider="local_hash")
        vector_a = provider._local_hash_embedding("SW-2100 AMOLED battery")
        vector_b = provider._local_hash_embedding("SW-2100 AMOLED battery")

        self.assertEqual(vector_a, vector_b)
        self.assertEqual(len(vector_a), 1536)
        self.assertGreater(sum(abs(value) for value in vector_a), 0)

    def test_source_path_for_chunked_and_direct_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            chunk = base / "Products" / "SW-2100-Flagship" / "1-122.md"
            direct = base / "Products" / "SW-1500-Sport.md"
            chunk.parent.mkdir(parents=True)
            direct.parent.mkdir(parents=True, exist_ok=True)
            chunk.write_text("chunk", encoding="utf-8")
            direct.write_text("direct", encoding="utf-8")

            self.assertEqual(
                source_path_for_chunk(chunk, base),
                "Products/SW-2100-Flagship.md",
            )
            self.assertEqual(
                source_path_for_chunk(direct, base),
                "Products/SW-1500-Sport.md",
            )

    def test_summary_for_chunk_matches_line_range(self):
        cache = {
            "Products/SW-2100-Flagship.md": [
                {"start": 1, "end": 122, "summary": "AMOLED 72h IP68"},
                {"start": 123, "end": 170, "summary": "GNSS workout price"},
            ]
        }

        self.assertEqual(
            summary_for_chunk(
                "Products/SW-2100-Flagship/1-122.md",
                "Products/SW-2100-Flagship.md",
                cache,
            ),
            "AMOLED 72h IP68",
        )

    def test_collect_chunk_documents_builds_search_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks = root / "Knowledge-Base-Chunks"
            summary_dir = root / "Knowledge-Base-File-Summary"
            chunk = chunks / "Products" / "SW-2100-Flagship" / "1-122.md"
            chunk.parent.mkdir(parents=True)
            summary_dir.mkdir()
            chunk.write_text("full chunk content", encoding="utf-8")
            (summary_dir / "summary_demo.json").write_text(
                json.dumps(
                    {
                        "Products/SW-2100-Flagship.md": [
                            {"start": 1, "end": 122, "summary": "AMOLED 72h IP68"}
                        ]
                    }
                ),
                encoding="utf-8",
            )

            docs = collect_chunk_documents(
                chunks_base=chunks,
                summary_file=summary_dir / "summary.txt",
            )

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].path, "Products/SW-2100-Flagship/1-122.md")
            self.assertEqual(docs[0].source_path, "Products/SW-2100-Flagship.md")
            self.assertIn("AMOLED 72h IP68", docs[0].search_text)
            self.assertIn("full chunk content", docs[0].search_text)
            self.assertEqual(docs[0].chunk_id, document_id_for_path(docs[0].path))
            self.assertEqual(docs[0].content_hash, content_hash("full chunk content"))

    def test_build_path_filter(self):
        expr = build_path_filter(
            ["2023-Market-Layout/", "Products/SW-1500-Sport.md"]
        )

        self.assertIn('path like "2023-Market-Layout/%"', expr)
        self.assertIn('source_path like "2023-Market-Layout/%"', expr)
        self.assertIn('path == "Products/SW-1500-Sport.md"', expr)
        self.assertIn('source_path == "Products/SW-1500-Sport.md"', expr)

    def test_build_path_filter_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            build_path_filter(["../secrets.md"])

    def test_parse_milvus_hits_accepts_dict_shape(self):
        hits = parse_milvus_hits(
            [
                [
                    {
                        "id": "abc",
                        "score": 0.42,
                        "entity": {
                            "path": "Products/SW-1500-Sport.md",
                            "source_path": "Products/SW-1500-Sport.md",
                        },
                    }
                ]
            ]
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].rank, 1)
        self.assertEqual(hits[0].path, "Products/SW-1500-Sport.md")
        self.assertEqual(hits[0].score, 0.42)


if __name__ == "__main__":
    unittest.main()
