from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.rag_service import RAGService


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            1.0 if "alpha" in lower else 0.0,
            1.0 if "beta" in lower else 0.0,
            1.0 if "gamma" in lower else 0.0,
            1.0,
        ]


class RAGServiceLifecycleTest(unittest.TestCase):
    def test_add_documents_is_idempotent_and_updates_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            doc_path = Path(temp_dir) / "product.txt"
            doc_path.write_text("alpha size guide", encoding="utf-8")

            rag = RAGService(
                data_dir=str(data_dir),
                embeddings=FakeEmbeddings(),
                default_use_rerank=False,
            )

            first = rag.add_documents([str(doc_path)])
            self.assertEqual(first["added_chunks"], 1)
            self.assertEqual(len(rag.list_documents()), 1)
            self.assertTrue((data_dir / "documents.json").exists())

            second = rag.add_documents([str(doc_path)])
            self.assertEqual(second["added_chunks"], 0)
            self.assertEqual(second["skipped_documents"], [str(doc_path)])
            self.assertEqual(len(rag.records), 1)

            doc_path.write_text("beta care guide", encoding="utf-8")
            third = rag.update_document(str(doc_path))

            self.assertEqual(third["added_chunks"], 1)
            self.assertEqual(third["deleted_chunks"], 1)
            self.assertEqual(third["updated_documents"], [str(doc_path)])
            self.assertEqual(rag.list_documents()[0]["version"], 2)
            self.assertEqual(len(rag.records), 1)
            self.assertIn("beta", rag.search("alpha", use_rerank=False)[0]["content"])

            deleted = rag.delete_document(str(doc_path))
            self.assertEqual(deleted["deleted_chunks"], 1)
            self.assertEqual(rag.list_documents(), [])
            self.assertEqual(rag.search("beta", use_rerank=False), [])

    def test_sync_documents_can_remove_missing_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            first_doc = Path(temp_dir) / "first.txt"
            second_doc = Path(temp_dir) / "second.txt"
            first_doc.write_text("alpha", encoding="utf-8")
            second_doc.write_text("gamma", encoding="utf-8")

            rag = RAGService(
                data_dir=str(data_dir),
                embeddings=FakeEmbeddings(),
                default_use_rerank=False,
            )
            rag.add_documents([str(first_doc), str(second_doc)])

            result = rag.sync_documents([str(first_doc)], remove_missing=True)

            self.assertEqual(result["removed_documents"], [str(second_doc)])
            self.assertEqual(
                [item["source"] for item in rag.list_documents()],
                [str(first_doc)],
            )
            self.assertEqual(len(rag.records), 1)


if __name__ == "__main__":
    unittest.main()
