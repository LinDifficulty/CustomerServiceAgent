from __future__ import annotations

import unittest
from typing import Any

from rag_server.query_rewrite import LLMQueryRewriter, QueryRewriteResult


class FakeRewriteModel:
    def invoke(self, messages: Any) -> Any:
        class Response:
            content = (
                '{"rewritten_query":"身高160cm体重47.5kg推荐什么尺码",'
                '"search_queries":["160cm 47.5kg 尺码推荐","身高160 体重95斤 尺码"],'
                '"notes":["保留数字约束"]}'
            )
        return Response()


class EmptyRewriteModel:
    def invoke(self, messages: Any) -> Any:
        class Response:
            content = '{"rewritten_query":"","search_queries":[],"notes":[]}'
        return Response()


class WrappedJsonModel:
    def invoke(self, messages: Any) -> Any:
        class Response:
            content = (
                "以下是改写结果：\n"
                '```json\n{"rewritten_query":"改写后的查询","search_queries":["查询1"],"notes":[]}\n```'
            )
        return Response()


class MultiPartContentModel:
    def invoke(self, messages: Any) -> Any:
        class Response:
            content = [
                {"text": '{"rewritten_query":"多段内容","search_queries":["q1"],"notes":[]}'}
            ]
        return Response()


class LLMQueryRewriterTests(unittest.TestCase):
    def test_rewrite_returns_structured_result(self) -> None:
        rewriter = LLMQueryRewriter(model=FakeRewriteModel())
        result = rewriter.rewrite("160cm 95斤穿什么码？")

        self.assertIsInstance(result, QueryRewriteResult)
        self.assertEqual(result.original_query, "160cm 95斤穿什么码？")
        self.assertIn("身高160cm", result.rewritten_query)
        self.assertGreater(len(result.search_queries), 0)
        self.assertIn(result.rewritten_query, result.search_queries)

    def test_rewrite_with_empty_response_falls_back_to_original(self) -> None:
        rewriter = LLMQueryRewriter(model=EmptyRewriteModel())
        result = rewriter.rewrite("原始问题")

        self.assertEqual(result.rewritten_query, "原始问题")
        self.assertIn("原始问题", result.search_queries)

    def test_rewrite_extracts_json_from_wrapped_text(self) -> None:
        rewriter = LLMQueryRewriter(model=WrappedJsonModel())
        result = rewriter.rewrite("测试查询")

        self.assertEqual(result.rewritten_query, "改写后的查询")

    def test_rewrite_handles_multi_part_content(self) -> None:
        rewriter = LLMQueryRewriter(model=MultiPartContentModel())
        result = rewriter.rewrite("测试查询")

        self.assertEqual(result.rewritten_query, "多段内容")

    def test_search_queries_deduplication(self) -> None:
        class DuplicateModel:
            def invoke(self, messages: Any) -> Any:
                class R:
                    content = (
                        '{"rewritten_query":"同一个查询",'
                        '"search_queries":["同一个查询","不同查询","同一个查询"],"notes":[]}'
                    )
                return R()

        rewriter = LLMQueryRewriter(model=DuplicateModel())
        result = rewriter.rewrite("原始")

        unique = list(dict.fromkeys(result.search_queries))
        self.assertEqual(result.search_queries, unique)

    def test_search_queries_capped_at_three(self) -> None:
        class ManyQueriesModel:
            def invoke(self, messages: Any) -> Any:
                class R:
                    content = (
                        '{"rewritten_query":"改写",'
                        '"search_queries":["q1","q2","q3","q4","q5"],"notes":[]}'
                    )
                return R()

        rewriter = LLMQueryRewriter(model=ManyQueriesModel())
        result = rewriter.rewrite("原始")

        self.assertLessEqual(len(result.search_queries), 3)


if __name__ == "__main__":
    unittest.main()
