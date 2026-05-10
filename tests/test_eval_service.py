"""测试检索评估服务，验证从 JSONL 数据集读取评测用例并计算命中率 (hit_rate) 和平均倒数排名 (MRR)。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.eval_service import evaluate_retrieval_dataset


class FakeRAGService:
    """模拟的 RAG 服务，根据查询返回预定义的检索结果。"""

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
    """检索评估服务单元测试：验证基于 JSONL 数据集的评测流水线，包括 source 匹配、doc_id 匹配和子串匹配。"""

    def test_evaluate_retrieval_dataset_summarizes_hits(self) -> None:
        """使用两份评测用例验证评估结果汇总：两条用例全部命中，hit_rate 和 MRR 均为 1.0。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 构造包含两条评测记录的 JSONL 数据集
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

        # 验证评测报告：2 条用例，全部命中
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["summary"]["hit_rate"], 1.0)
        self.assertEqual(report["summary"]["mrr"], 1.0)
        self.assertTrue(all(item["hit"] for item in report["cases"]))


if __name__ == "__main__":
    unittest.main()
