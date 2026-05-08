from __future__ import annotations

import asyncio
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from rag_server.cli import build_agent
from rag_server.reflection_service import (
    format_retrieval_results,
    parse_reflection_result,
)


class FakeReflectionModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="这款商品有现货，可以直接拍。")
        if self.calls == 2:
            return AIMessage(
                content=(
                    '{"has_hallucination": true, '
                    '"needs_more_evidence": true, '
                    '"reason": "库存承诺没有证据支持", '
                    '"search_query": "商品 库存", '
                    '"correction_guidance": "删除库存承诺"}'
                )
            )
        return AIMessage(content="目前知识库没有库存信息，我无法确认是否有现货。")


class FakeRAG:
    def search(self, query: str, top_k: int = 3, candidate_top_k: int | None = None):
        return [
            {
                "score": 0.9,
                "source": "docs/商品说明.txt",
                "content": "知识库包含尺码和洗护信息，但没有库存信息。",
                "metadata": {"chunk_index": 0},
                "retrieval_mode": "hybrid",
            }
        ]


class ReflectionServiceTests(unittest.TestCase):
    def test_parse_reflection_result_from_json_block(self) -> None:
        result = parse_reflection_result(
            """
            审查如下：
            {
              "has_hallucination": true,
              "needs_more_evidence": true,
              "reason": "尺码建议没有证据支持",
              "search_query": "160cm 95斤 尺码",
              "correction_guidance": "补充检索后再给建议"
            }
            """
        )

        self.assertTrue(result.has_hallucination)
        self.assertTrue(result.needs_more_evidence)
        self.assertTrue(result.needs_revision)
        self.assertEqual(result.search_query, "160cm 95斤 尺码")

    def test_parse_reflection_result_defaults_to_no_revision(self) -> None:
        result = parse_reflection_result("不是 JSON")

        self.assertFalse(result.has_hallucination)
        self.assertFalse(result.needs_more_evidence)
        self.assertFalse(result.needs_revision)

    def test_format_retrieval_results(self) -> None:
        formatted = format_retrieval_results(
            [
                {
                    "source": "docs/尺码推荐.txt",
                    "content": "160cm、95斤建议 S 码。",
                }
            ]
        )

        self.assertIn("片段1", formatted)
        self.assertIn("docs/尺码推荐.txt", formatted)
        self.assertIn("建议 S 码", formatted)

    def test_agent_reflection_revises_final_answer(self) -> None:
        async def run_case() -> str:
            app, _, _ = build_agent(
                FakeRAG(),
                query_rewrite_mode="off",
                skills_enabled=False,
                memory_service=None,
                memory_extractor=None,
                agent_model=FakeReflectionModel(),
                reflection_enabled=True,
            )
            result = await app.ainvoke(
                {
                    "messages": [HumanMessage(content="这款现在有货吗？")],
                    "user_id": "user",
                }
            )
            return result["messages"][-1].content

        final_answer = asyncio.run(run_case())

        self.assertIn("无法确认", final_answer)
        self.assertNotIn("可以直接拍", final_answer)


if __name__ == "__main__":
    unittest.main()
