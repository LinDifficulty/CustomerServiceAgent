from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.eval_service import evaluate_retrieval_dataset


class FakeRAGService:
    def search(self, query: str, **_: object) -> list[dict]:
        if query == "尺码怎么选":
            return [
                {
                    "score": 0.9,
                    "source": "docs/尺码推荐.txt",
                    "doc_id": "size-doc",
                    "content": "建议尺码M，喜欢宽松可选L。",
                    "metadata": {"chunk_index": 0},
                    "retrieval_mode": "hybrid",
                }
            ]
        return [
            {
                "score": 0.8,
                "source": "docs/颜色选择.txt",
                "doc_id": "color-doc",
                "content": "正式场合建议选择黑白灰、藏蓝等稳重颜色。",
                "metadata": {"chunk_index": 1},
                "retrieval_mode": "hybrid",
            }
        ]


class EvalServiceTests(unittest.TestCase):
    def test_evaluate_retrieval_dataset_summarizes_hits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "retrieval_eval.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        (
                            '{"id":"size","query":"尺码怎么选",'
                            '"expected_sources":["尺码推荐.txt"],'
                            '"expected_substrings":["建议尺码M"]}'
                        ),
                        (
                            '{"id":"color","query":"面试穿什么颜色",'
                            '"expected_doc_ids":["color-doc"],'
                            '"expected_substrings":["正式场合"]}'
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            report = evaluate_retrieval_dataset(
                FakeRAGService(),
                dataset_path,
                top_k=1,
            )

        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["summary"]["hit_rate"], 1.0)
        self.assertEqual(report["summary"]["mrr"], 1.0)
        self.assertTrue(all(item["hit"] for item in report["cases"]))


if __name__ == "__main__":
    unittest.main()
