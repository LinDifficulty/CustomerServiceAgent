"""测试 LLM 查询改写器（LLMQueryRewriter）的各种场景。

包含以下测试场景：
- 同步/异步查询改写返回 QueryRewriteResult 结构
- LLM 返回空结果时的回退到原始查询
- JSON 被包裹在文本或代码块中的提取
- 多段内容（multi-part content）的处理
- search_queries 去重和数量上限（最多 3 条）
- 重复查询的缓存命中
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from rag_server.cache_service import InMemoryJsonCache
from rag_server.query_rewrite import LLMQueryRewriter, QueryRewriteResult

# --- 测试用 Fake 模型 ---


class FakeRewriteModel:
    """参数化假模型：返回指定的 content 作为 LLM 响应。"""

    def __init__(self, content: Any) -> None:
        self._content = content

    def invoke(self, messages: Any) -> Any:
        class Response:
            content = self._content

        return Response()


# 常用预置实例
VALID_REWRITE_JSON = (
    '{"rewritten_query":"身高160cm体重47.5kg推荐什么尺码",'
    '"search_queries":["160cm 47.5kg 尺码推荐","身高160 体重95斤 尺码"],'
    '"notes":["保留数字约束"]}'
)
EMPTY_REWRITE_JSON = '{"rewritten_query":"","search_queries":[],"notes":[]}'
WRAPPED_JSON = (
    '以下是改写结果：\n```json\n{"rewritten_query":"改写后的查询","search_queries":["查询1"],"notes":[]}\n```'
)
MULTI_PART_CONTENT = [{"text": '{"rewritten_query":"多段内容","search_queries":["q1"],"notes":[]}'}]


class LLMQueryRewriterTests(unittest.TestCase):
    """测试 LLMQueryRewriter 的改写、解析、去重和缓存功能。"""

    def test_rewrite_returns_structured_result(self) -> None:
        """验证同步改写返回 QueryRewriteResult 结构，且改写后的查询包含在搜索查询列表中。"""
        rewriter = LLMQueryRewriter(model=FakeRewriteModel(VALID_REWRITE_JSON))
        result = rewriter.rewrite("160cm 95斤穿什么码？")

        self.assertIsInstance(result, QueryRewriteResult)
        self.assertEqual(result.original_query, "160cm 95斤穿什么码？")
        self.assertIn("身高160cm", result.rewritten_query)
        self.assertGreater(len(result.search_queries), 0)
        self.assertIn(result.rewritten_query, result.search_queries)

    def test_arewrite_returns_structured_result(self) -> None:
        """验证异步改写同样返回正确的 QueryRewriteResult 结构。"""

        async def run_case() -> QueryRewriteResult:
            rewriter = LLMQueryRewriter(model=FakeRewriteModel(VALID_REWRITE_JSON))
            return await rewriter.arewrite("160cm 95斤穿什么码？")

        result = asyncio.run(run_case())

        self.assertIsInstance(result, QueryRewriteResult)
        self.assertEqual(result.original_query, "160cm 95斤穿什么码？")
        self.assertIn("身高160cm", result.rewritten_query)

    def test_rewrite_with_empty_response_falls_back_to_original(self) -> None:
        """当 LLM 返回空结果时，回退到原始查询，原始查询应出现在 search_queries 中。"""
        rewriter = LLMQueryRewriter(model=FakeRewriteModel(EMPTY_REWRITE_JSON))
        result = rewriter.rewrite("原始问题")

        self.assertEqual(result.rewritten_query, "原始问题")
        self.assertIn("原始问题", result.search_queries)

    def test_rewrite_extracts_json_from_wrapped_text(self) -> None:
        """验证能从包裹在 markdown 代码块中的 JSON 正确提取改写结果。"""
        rewriter = LLMQueryRewriter(model=FakeRewriteModel(WRAPPED_JSON))
        result = rewriter.rewrite("测试查询")

        self.assertEqual(result.rewritten_query, "改写后的查询")

    def test_rewrite_handles_multi_part_content(self) -> None:
        """验证能处理 LLM 返回的多段内容格式（content 为列表）。"""
        rewriter = LLMQueryRewriter(model=FakeRewriteModel(MULTI_PART_CONTENT))
        result = rewriter.rewrite("测试查询")

        self.assertEqual(result.rewritten_query, "多段内容")

    def test_search_queries_deduplication(self) -> None:
        """验证搜索查询列表中的重复项被去重，保持原有顺序。"""
        dup_content = (
            '{"rewritten_query":"同一个查询","search_queries":["同一个查询","不同查询","同一个查询"],"notes":[]}'
        )
        rewriter = LLMQueryRewriter(model=FakeRewriteModel(dup_content))
        result = rewriter.rewrite("原始")

        unique = list(dict.fromkeys(result.search_queries))
        self.assertEqual(result.search_queries, unique)

    def test_search_queries_capped_at_three(self) -> None:
        """验证搜索查询数量上限为 3 条，超过部分被截断。"""
        many_content = '{"rewritten_query":"改写","search_queries":["q1","q2","q3","q4","q5"],"notes":[]}'
        rewriter = LLMQueryRewriter(model=FakeRewriteModel(many_content))
        result = rewriter.rewrite("原始")

        self.assertLessEqual(len(result.search_queries), 3)

    def test_rewrite_uses_cache_for_repeated_query(self) -> None:
        """验证重复查询命中缓存，LLM 模型仅被调用一次。"""
        cache_content = '{"rewritten_query":"缓存查询","search_queries":["缓存查询"],"notes":[]}'

        class CountingModel(FakeRewriteModel):
            calls = 0

            def invoke(self, messages: Any) -> Any:
                self.calls += 1
                return super().invoke(messages)

        model = CountingModel(cache_content)
        cache = InMemoryJsonCache(namespace="rewrite-test")
        rewriter = LLMQueryRewriter(model=model, cache=cache)

        first = rewriter.rewrite("重复问题")
        second = rewriter.rewrite("重复问题")

        self.assertEqual(first.rewritten_query, second.rewritten_query)
        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
