"""测试记忆服务（MemoryService）的全生命周期。

包含以下测试场景：
- MemoryService 的记忆增删改查、批量添加、列表、搜索、分层搜索
- 记忆搜索的缓存机制
- 用户隔离（不同用户的记忆不会互相干扰）
- 重要性钳位、非法记忆类型回退、空内容拒绝
- memory_layer_for_type 的类型到分层映射
- LLMMemoryExtractor 的同步/异步记忆提取和 JSON 解析
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from typing import Any

from rag_server.cache_service import InMemoryJsonCache
from rag_server.memory_service import (
    MEMORY_LAYERS,
    MEMORY_TYPES,
    ExtractedMemory,
    LLMMemoryExtractor,
    MemoryService,
    memory_layer_for_type,
)


# --- 测试用 Fake 对象 ---

class FakeEmbeddings:
    """模拟嵌入模型：根据文本关键词返回确定性向量。"""

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        if "尺码" in text or "身高" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "颜色" in text or "色" in text:
            return [0.0, 1.0, 0.0, 0.0]
        if "洗涤" in text or "洗" in text:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]


class CountingEmbeddings(FakeEmbeddings):
    """带调用计数的嵌入模型，用于验证缓存是否生效。"""

    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += len(texts)
        return super().embed_documents(texts)


class FakeModel:
    """模拟 LLM 模型：返回空的记忆列表。"""

    def invoke(self, messages: Any) -> Any:
        class Response:
            content = '{"memories":[]}'
        return Response()


class MemoryServiceTests(unittest.TestCase):
    """测试 MemoryService 的核心 CRUD 操作和记忆管理功能。"""

    def test_add_and_get_memory(self) -> None:
        """测试添加记忆后能正确读取，验证 content、memory_type、memory_layer、user_id 字段。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            record = ms.add_memory("u1", "喜欢宽松版型", memory_type="preference")
            fetched = ms.get_memory(record["id"])
            ms.close()

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["content"], "喜欢宽松版型")
        self.assertEqual(fetched["memory_type"], "preference")
        self.assertEqual(fetched["memory_layer"], "profile")
        self.assertEqual(fetched["user_id"], "u1")

    def test_add_memories_batch(self) -> None:
        """测试批量添加记忆，空内容的记忆应被过滤掉。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            records = ms.add_memories(
                "u1",
                [
                    {"content": "身高170", "memory_type": "profile", "importance": 0.9},
                    {"content": "体重60kg", "memory_type": "profile", "importance": 0.9},
                    {"content": "", "memory_type": "profile"},
                ],
            )
            ms.close()

        self.assertEqual(len(records), 2)

    def test_list_memories(self) -> None:
        """测试按用户列出所有记忆，验证只返回该用户的记忆。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            ms.add_memory("u1", "偏好A", memory_type="preference")
            ms.add_memory("u1", "偏好B", memory_type="preference")
            ms.add_memory("u2", "偏好C", memory_type="preference")
            result = ms.list_memories("u1")
            ms.close()

        self.assertEqual(len(result), 2)

    def test_search_memory(self) -> None:
        """测试向量搜索记忆，验证返回结果包含 score 字段。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            ms.add_memory("u1", "身高175cm", memory_type="profile")
            ms.add_memory("u1", "喜欢红色", memory_type="preference")
            results = ms.search_memory("u1", "尺码 身高", top_k=2)
            ms.close()

        self.assertGreater(len(results), 0)
        self.assertIn("score", results[0])

    def test_search_memory_uses_cache_for_repeated_query(self) -> None:
        """测试重复查询使用缓存：第二次搜索不触发额外的嵌入计算。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = CountingEmbeddings()
            ms = MemoryService(
                data_dir=temp_dir,
                embeddings=embeddings,
                cache=InMemoryJsonCache(namespace="memory-test"),
            )
            ms.add_memory("u1", "身高175cm", memory_type="profile")
            calls_after_add = embeddings.document_calls

            first = ms.search_memory("u1", "尺码 身高", top_k=2)
            calls_after_first = embeddings.document_calls
            second = ms.search_memory("u1", "尺码 身高", top_k=2)
            ms.close()

        self.assertEqual(len(second), len(first))
        self.assertEqual(calls_after_first, calls_after_add + 1)
        self.assertEqual(embeddings.document_calls, calls_after_first)

    def test_search_memory_layers(self) -> None:
        """测试按记忆分层搜索，返回结果应包含 profile/episode/procedure 三个层。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            ms.add_memory("u1", "身高175cm", memory_type="profile")
            ms.add_memory("u1", "上次买了M码", memory_type="episode")
            ms.add_memory("u1", "每次先查尺码表", memory_type="procedure")
            layered = ms.search_memory_layers("u1", "身高 尺码")
            ms.close()

        self.assertIn("profile", layered)
        self.assertIn("episode", layered)
        self.assertIn("procedure", layered)

    def test_asearch_memory_layers(self) -> None:
        """测试异步分层搜索，与同步版本结果一致。"""
        async def run_case() -> dict[str, list[dict]]:
            with tempfile.TemporaryDirectory() as temp_dir:
                ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
                ms.add_memory("u1", "身高175cm", memory_type="profile")
                ms.add_memory("u1", "上次买了M码", memory_type="episode")
                ms.add_memory("u1", "每次先查尺码表", memory_type="procedure")
                layered = await ms.asearch_memory_layers("u1", "身高 尺码")
                ms.close()
                return layered

        layered = asyncio.run(run_case())

        self.assertIn("profile", layered)
        self.assertIn("episode", layered)
        self.assertIn("procedure", layered)

    def test_forget_memory(self) -> None:
        """测试软删除记忆：删除后 get_memory 返回 None。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            record = ms.add_memory("u1", "要删的记忆")
            deleted = ms.forget_memory(record["id"])
            fetched = ms.get_memory(record["id"])
            ms.close()

        self.assertTrue(deleted)
        self.assertIsNone(fetched)

    def test_forget_nonexistent_memory(self) -> None:
        """测试删除不存在的记忆时返回 False。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            deleted = ms.forget_memory("no-such-id")
            ms.close()

        self.assertFalse(deleted)

    def test_clear_user_memory(self) -> None:
        """测试清除用户所有记忆，且不影响其他用户的记忆。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            ms.add_memory("u1", "记忆1")
            ms.add_memory("u1", "记忆2")
            ms.add_memory("u2", "记忆3")
            count = ms.clear_user_memory("u1")
            remaining_u1 = ms.list_memories("u1")
            remaining_u2 = ms.list_memories("u2")
            ms.close()

        self.assertEqual(count, 2)
        self.assertEqual(len(remaining_u1), 0)
        self.assertEqual(len(remaining_u2), 1)

    def test_user_isolation(self) -> None:
        """测试用户隔离：用户 u2 搜索不到用户 u1 的记忆。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            ms.add_memory("u1", "身高175cm", memory_type="profile")
            results_u2 = ms.search_memory("u2", "身高", top_k=5)
            ms.close()

        self.assertEqual(len(results_u2), 0)

    def test_importance_clamped(self) -> None:
        """测试重要性值被钳位到 [0.0, 1.0] 范围内。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            r1 = ms.add_memory("u1", "高重要性", importance=2.0)
            r2 = ms.add_memory("u1", "负重要性", importance=-0.5)
            ms.close()

        self.assertEqual(r1["importance"], 1.0)
        self.assertEqual(r2["importance"], 0.0)

    def test_invalid_memory_type_defaults_to_preference(self) -> None:
        """测试非法记忆类型自动回退为 "preference"。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            record = ms.add_memory("u1", "测试", memory_type="invalid_type")
            ms.close()

        self.assertEqual(record["memory_type"], "preference")

    def test_empty_content_raises(self) -> None:
        """测试空内容记忆添加时抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ms = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            with self.assertRaises(ValueError):
                ms.add_memory("u1", "")
            ms.close()


class MemoryLayerTests(unittest.TestCase):
    """测试记忆类型到分层的映射关系。"""

    def test_layer_mapping(self) -> None:
        """验证各类记忆类型映射到正确的分层：profile 类→profile，episode→episode，procedure→procedure，未知→profile。"""
        self.assertEqual(memory_layer_for_type("profile"), "profile")
        self.assertEqual(memory_layer_for_type("preference"), "profile")
        self.assertEqual(memory_layer_for_type("constraint"), "profile")
        self.assertEqual(memory_layer_for_type("instruction"), "profile")
        self.assertEqual(memory_layer_for_type("episode"), "episode")
        self.assertEqual(memory_layer_for_type("procedure"), "procedure")
        self.assertEqual(memory_layer_for_type("unknown"), "profile")

    def test_all_types_have_a_layer(self) -> None:
        """验证所有已定义的记忆类型都有对应的合法分层。"""
        for memory_type in MEMORY_TYPES:
            layer = memory_layer_for_type(memory_type)
            self.assertIn(layer, MEMORY_LAYERS)


class LLMMemoryExtractorTests(unittest.TestCase):
    """测试 LLM 记忆提取器（LLMMemoryExtractor）的同步和异步提取。"""

    def test_extract_returns_empty_list_for_no_memories(self) -> None:
        """当 LLM 返回空记忆列表时，extract 返回空列表。"""
        extractor = LLMMemoryExtractor(model=FakeModel())
        result = extractor.extract(
            user_message="你好",
            assistant_message="您好！",
        )
        self.assertEqual(result, [])

    def test_aextract_returns_empty_list_for_no_memories(self) -> None:
        """当 LLM 返回空记忆列表时，异步 aextract 也返回空列表。"""
        async def run_case() -> list[ExtractedMemory]:
            extractor = LLMMemoryExtractor(model=FakeModel())
            return await extractor.aextract(
                user_message="你好",
                assistant_message="您好！",
            )

        result = asyncio.run(run_case())

        self.assertEqual(result, [])

    def test_extract_parses_valid_memories(self) -> None:
        """验证 LLM 返回的有效 JSON 记忆能正确解析为 ExtractedMemory 对象。"""
        class MemoryModel:
            def invoke(self, messages: Any) -> Any:
                class R:
                    content = (
                        '{"memories":[{"content":"用户身高175cm","memory_type":"profile",'
                        '"importance":0.9,"expires_at":null}]}'
                    )
                return R()

        extractor = LLMMemoryExtractor(model=MemoryModel())
        result = extractor.extract(
            user_message="我身高175cm",
            assistant_message="好的，已记录。",
        )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ExtractedMemory)
        self.assertEqual(result[0].content, "用户身高175cm")
        self.assertEqual(result[0].memory_type, "profile")
        self.assertAlmostEqual(result[0].importance, 0.9)


if __name__ == "__main__":
    unittest.main()
