from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.eval_service import evaluate_retrieval_dataset
from rag_server.rag_service import RAGService
from rag_server.trace_service import TraceRecorder, load_trace


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            1.0 if "cotton" in lower else 0.0,
            1.0 if "silk" in lower else 0.0,
            1.0 if "wool" in lower else 0.0,
            1.0,
        ]


class EvalTraceTest(unittest.TestCase):
    def test_trace_recorder_captures_rag_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = TraceRecorder(trace_dir=Path(temp_dir) / "traces")
            doc_path = Path(temp_dir) / "care.txt"
            doc_path.write_text("cotton can be washed gently", encoding="utf-8")
            rag = RAGService(
                data_dir=str(Path(temp_dir) / "data"),
                embeddings=FakeEmbeddings(),
                default_use_rerank=False,
                trace_recorder=trace,
            )

            rag.add_documents([str(doc_path)])
            rag.search("cotton wash", use_rerank=False)

            records = load_trace(trace.path)
            names = [item["name"] for item in records]

            self.assertIn("rag.upsert_documents", names)
            self.assertIn("rag.search_by_hybrid", names)
            self.assertIn("rag.search", names)

    def test_retrieval_eval_reports_hits_and_traces_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "care.txt"
            doc_path.write_text(
                "silk should be hand washed with mild detergent",
                encoding="utf-8",
            )
            dataset_path = Path(temp_dir) / "retrieval_eval.jsonl"
            dataset_path.write_text(
                (
                    '{"id":"silk","query":"silk detergent",'
                    f'"expected_sources":["{doc_path.name}"],'
                    '"expected_substrings":["mild detergent"]}\n'
                ),
                encoding="utf-8",
            )
            trace = TraceRecorder(trace_dir=Path(temp_dir) / "traces")
            rag = RAGService(
                data_dir=str(Path(temp_dir) / "data"),
                embeddings=FakeEmbeddings(),
                default_use_rerank=False,
                trace_recorder=trace,
            )
            rag.add_documents([str(doc_path)])

            report = evaluate_retrieval_dataset(
                rag,
                dataset_path,
                top_k=1,
                use_rerank=False,
                trace_recorder=trace,
            )

            self.assertEqual(report["case_count"], 1)
            self.assertEqual(report["summary"]["hit_rate"], 1.0)
            self.assertEqual(report["summary"]["mrr"], 1.0)
            names = [item["name"] for item in load_trace(trace.path)]
            self.assertIn("eval.retrieval_case", names)
            self.assertIn("eval.retrieval_summary", names)


if __name__ == "__main__":
    unittest.main()
