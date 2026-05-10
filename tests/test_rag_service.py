"""RAGService 单元测试。

测试文档入库、FAISS 向量索引构建、BM25 关键词召回、
多向量检索（关键词向量 + 摘要向量）、父子分块（parent-child chunking）、
幂等入库（内容哈希去重）、增量入库、跨编码器默认禁用、
以及查询向量缓存等核心 RAG 功能。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rag_server.cache_service import InMemoryJsonCache
from rag_server.rag_service import RAGService


# 用于测试的模拟嵌入类——返回固定向量，避免调用真实 API
class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


# 计数嵌入类——记录 embed_query 调用次数，用于验证查询缓存是否生效
class CountingQueryEmbeddings(FakeEmbeddings):
    def __init__(self) -> None:
        self.query_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(text)


# 多向量嵌入类——根据文本内容返回不同的向量，模拟关键词向量和摘要向量的多向量搜索场景
class MultiVectorEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        if "靛蓝独有词" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        if text.startswith("关键词:") and (
            "靛蓝独有词" in text or ("靛蓝" in text and "独有" in text)
        ):
            return [1.0, 0.0, 0.0]
        if text.startswith("摘要:"):
            return [0.0, 0.0, 1.0]
        return [0.0, 1.0, 0.0]


class EmbeddingShouldNotBeCalled(Exception):
    """Raised to verify empty RAGService init does not call embedding methods."""


class ExplodingEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingShouldNotBeCalled("empty RAGService init should not embed a probe query")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingShouldNotBeCalled("empty RAGService init should not embed documents")


class RAGServiceTests(unittest.TestCase):
    # 测试 jieba 分词初始化日志已被静默，确保终端没有多余输出
    def test_jieba_initialization_logs_are_silenced(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from rag_server.rag_service import jieba\n"
                    "list(jieba.lcut_for_search('中文关键词测试'))\n"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Building prefix dict", result.stderr)
        self.assertNotIn("Loading model from cache", result.stderr)
        self.assertNotIn("Prefix dict has been built successfully", result.stderr)

    # 验证跨编码器默认不启用，避免首次运行时自动下载大模型
    def test_cross_encoder_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rag = RAGService(data_dir=temp_dir, embeddings=FakeEmbeddings())

        self.assertFalse(rag.default_use_rerank)

    # 验证空的 RAGService 初始化不会触发嵌入调用（无文档则不建索引）
    def test_empty_init_does_not_call_embedding_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rag = RAGService(data_dir=temp_dir, embeddings=ExplodingEmbeddings())

        self.assertEqual(rag.index.ntotal, 0)

    # 验证空初始化后添加文档能正确构建 FAISS 索引（索引维度 = 3，总向量数 = 3）
    def test_add_documents_after_empty_init_builds_real_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "doc.txt"
            doc_path.write_text("一段可检索的测试知识", encoding="utf-8")
            rag = RAGService(data_dir=temp_dir, embeddings=FakeEmbeddings())

            result = rag.add_documents([str(doc_path)])

        self.assertEqual(result["added_chunks"], 1)
        self.assertEqual(rag.index.d, 3)
        self.assertEqual(rag.index.ntotal, 3)

    # 验证多向量检索：包含关键词的查询优先使用关键词向量（best_vector_type=keyword）进行召回
    def test_multi_vector_retrieval_uses_keyword_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir()
            keyword_doc = docs_dir / "keyword.txt"
            other_doc = docs_dir / "other.txt"
            keyword_doc.write_text(
                "普通描述里包含靛蓝独有词，用来测试关键词向量召回。",
                encoding="utf-8",
            )
            other_doc.write_text("完全无关的商品知识片段。", encoding="utf-8")
            rag = RAGService(data_dir=str(data_dir), embeddings=MultiVectorEmbeddings())

            rag.add_documents([str(keyword_doc), str(other_doc)])
            results = rag.search_by_vector("靛蓝独有词", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(Path(results[0]["source"]).name, "keyword.txt")
        self.assertEqual(results[0]["best_vector_type"], "keyword")
        self.assertIn("keyword", results[0]["matched_vector_types"])
        self.assertGreater(results[0]["multi_vector_scores"]["keyword"], 0.99)

    # 验证异步搜索 asearch 的结果与同步搜索 search 的结果一致
    def test_asearch_matches_search_result_source(self) -> None:
        async def run_case() -> tuple[list[dict], list[dict]]:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_dir = Path(temp_dir) / "data"
                doc_path = Path(temp_dir) / "doc.txt"
                doc_path.write_text("一段可检索的尺码知识", encoding="utf-8")
                rag = RAGService(data_dir=str(data_dir), embeddings=FakeEmbeddings())
                rag.add_documents([str(doc_path)])

                sync_results = rag.search("尺码", top_k=1)
                async_results = await rag.asearch("尺码", top_k=1)
                return sync_results, async_results

        sync_results, async_results = asyncio.run(run_case())

        self.assertEqual(len(async_results), len(sync_results))
        self.assertEqual(async_results[0]["source"], sync_results[0]["source"])

    # 验证父子分块：子块被索引、搜索返回父块，且子块内容含在 child_content 中
    def test_parent_child_chunking_indexes_children_and_returns_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "doc.txt"
            doc_path.write_text(
                "Parent intro context before the key term. "
                "needle child detail. "
                "Trailing parent context after the key term.",
                encoding="utf-8",
            )
            rag = RAGService(
                data_dir=temp_dir,
                embeddings=FakeEmbeddings(),
                chunk_size=24,
                chunk_overlap=0,
                parent_chunk_size=160,
                parent_chunk_overlap=0,
            )

            result = rag.add_documents([str(doc_path)])
            results = rag.search_by_bm25("needle", top_k=1)

        self.assertEqual(result["added_parent_chunks"], 1)
        self.assertGreater(result["added_chunks"], result["added_parent_chunks"])
        self.assertEqual(len(results), 1)
        self.assertIn("Trailing parent context", results[0]["content"])
        self.assertIn("needle", results[0]["child_content"])
        self.assertEqual(results[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(results[0]["metadata"]["parent_index"], 0)
        self.assertIn("child_chunk_id", results[0]["metadata"])

    # 验证父子分块去重：多个子块命中同一个父块时，结果按父块去重只返回一条
    def test_parent_child_results_are_deduplicated_by_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "doc.txt"
            doc_path.write_text(
                "repeat first child detail. "
                "repeat second child detail. "
                "repeat third child detail. "
                "same parent tail.",
                encoding="utf-8",
            )
            rag = RAGService(
                data_dir=temp_dir,
                embeddings=FakeEmbeddings(),
                chunk_size=28,
                chunk_overlap=0,
                parent_chunk_size=180,
                parent_chunk_overlap=0,
            )

            rag.add_documents([str(doc_path)])
            results = rag.search_by_bm25("repeat", top_k=5)

        self.assertEqual(len(results), 1)
        self.assertIn("same parent tail", results[0]["content"])

    # 验证旧格式 metadata.json 能自动迁移重建 FAISS 索引和 documents.json
    def test_legacy_metadata_rebuilds_persistent_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            metadata_path = data_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "source": "docs/example.txt",
                            "content": "示例知识片段",
                            "metadata": {"chunk_index": 0},
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            RAGService(data_dir=temp_dir, embeddings=FakeEmbeddings())

            self.assertTrue((data_dir / "faiss.index").exists())
            documents_payload = json.loads(
                (data_dir / "documents.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(documents_payload["documents"]), 1)

    # 验证幂等入库：相同内容不重复入库；内容变更后自动更新并重新索引
    def test_add_documents_is_idempotent_and_updates_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "doc.txt"
            doc_path.write_text("原始内容", encoding="utf-8")
            rag = RAGService(data_dir=temp_dir, embeddings=FakeEmbeddings())

            first = rag.add_documents([str(doc_path)])
            self.assertEqual(first["added_chunks"], 1)

            second = rag.add_documents([str(doc_path)])
            self.assertEqual(second["added_chunks"], 0)
            self.assertEqual(second["skipped_documents"], [str(doc_path)])

            doc_path.write_text("修改后的内容", encoding="utf-8")
            third = rag.add_documents([str(doc_path)])
            self.assertEqual(third["added_chunks"], 1)
            self.assertIn(str(doc_path), third["updated_documents"])

    # 验证增量入库：新增文档仅嵌入新增块，不会触发全量重建
    def test_incremental_add_extends_index_without_full_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            doc1 = Path(temp_dir) / "doc1.txt"
            doc2 = Path(temp_dir) / "doc2.txt"
            doc1.write_text("第一篇文档内容", encoding="utf-8")
            doc2.write_text("第二篇文档内容", encoding="utf-8")

            embed_calls: list[list[str]] = []
            original_embed = FakeEmbeddings.embed_documents

            def tracking_embed(self, texts):
                embed_calls.append(list(texts))
                return original_embed(self, texts)

            FakeEmbeddings.embed_documents = tracking_embed
            try:
                rag = RAGService(data_dir=temp_dir, embeddings=FakeEmbeddings())
                rag.add_documents([str(doc1)])
                index_size_after_first = rag.index.ntotal

                embed_calls.clear()
                rag.add_documents([str(doc2)])
                calls_after_second = len(embed_calls)
            finally:
                FakeEmbeddings.embed_documents = original_embed

            self.assertGreater(rag.index.ntotal, index_size_after_first)
            self.assertEqual(calls_after_second, 1)

    # 验证查询向量缓存：相同查询词的向量搜索命中缓存，降低 embedding API 调用次数
    def test_query_embedding_uses_cache_for_repeated_vector_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = Path(temp_dir) / "doc.txt"
            doc_path.write_text("一段可检索的尺码知识", encoding="utf-8")
            embeddings = CountingQueryEmbeddings()
            rag = RAGService(
                data_dir=temp_dir,
                embeddings=embeddings,
                cache=InMemoryJsonCache(namespace="rag-test"),
            )
            rag.add_documents([str(doc_path)])

            rag.search_by_vector("尺码", top_k=1)
            rag.search_by_vector("尺码", top_k=1)

        self.assertEqual(embeddings.query_calls, 1)


if __name__ == "__main__":
    unittest.main()
