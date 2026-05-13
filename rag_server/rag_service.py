"""
RAG 核心检索引擎模块。

本模块是本项目的核心，实现了完整的 RAG（Retrieval-Augmented Generation）检索管线：
1. 多向量 FAISS 嵌入检索 —— 支持摘要/关键词/语义三种向量类型的加权融合
2. BM25Plus 关键词检索 —— 基于 jieba 中文分词，与向量召回互补
3. 混合融合检索 —— 向量召回 + BM25 的加权混合排序
4. CrossEncoder 重排序 —— 对候选结果进行精细排序
5. 父子分块（Parent-Child Chunking）—— 小粒度子块用于检索，大粒度父块用于展示
6. 文档增删改同步 —— 支持增量的向量索引扩展，避免全量重建
7. 多层缓存 —— 嵌入缓存、检索缓存、重排序缓存，大幅降低 API 调用成本

检索流程：用户查询 → 多向量召回 + BM25 召回 → 加权融合 → (可选) CrossEncoder 重排序 → 返回 top_k 结果
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import faiss  # Facebook AI Similarity Search，高效向量相似度检索库
import numpy as np  # 数值计算，向量/矩阵操作
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归字符分块器，按自然语义边界切分文本
from pypdf import PdfReader  # PDF 文档解析
from rank_bm25 import BM25Plus  # BM25+ 关键词检索算法，相比 BM25Okapi 在小语料下更稳定

from .cache_service import CacheTTLs, JsonCache, read_cached_list, stable_cache_digest, write_cached_list
from .model_factory import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_PROVIDER,
    create_embeddings,
    create_reranker,
    model_config_fingerprint,  # 根据模型配置生成指纹 Hash，用于缓存键和变更检测
)
from .trace_service import TraceRecorder, summarize_result
from .utils import cache_key_or_none, normalize_vector_score, utc_now

# jieba 分词库初始化时会触发 pkg_resources 弃用警告，这里忽略该噪音日志
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import jieba  # 中文分词库，用于 BM25 关键词检索和关键词提取

# 设置 jieba 日志级别为 WARNING，避免终端输出过多分词日志
jieba.setLogLevel(logging.WARNING)

# ── 文档与分块配置 ───────────────────────────────────────────────
# 支持的文档格式集合，仅处理这些扩展名的文件。
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# documents.json 清单文件的版本号，用于未来兼容性升级。
DOCUMENTS_MANIFEST_VERSION = 1

# 分块策略标识：parent_child 表示父子分块（子块用于检索，父块用于展示上下文）。
CHUNKING_STRATEGY = "parent_child"

# 候选召回放大因子：向量/BM25 检索时，先召回 top_k * OVERFETCH 个候选，
# 再经过去重和融合后截断为 top_k，避免因父块去重导致结果不足。
PARENT_CHILD_OVERFETCH_FACTOR = 5

# ── 多向量嵌入配置 ───────────────────────────────────────────────
# 多向量策略标识：summary（摘要向量）、keyword（关键词向量）、semantic（语义向量）。
MULTI_VECTOR_STRATEGY = "summary_keyword_semantic_v1"

# 三种向量类型元组，作为索引构建和检索的统一遍历顺序。
MULTI_VECTOR_TYPES = ("summary", "keyword", "semantic")

# 多向量融合的默认权重：摘要 25%、关键词 25%、语义 50%。
# 语义向量权重最高，因为它保留了最完整的文本信息。
DEFAULT_MULTI_VECTOR_WEIGHTS = {
    "summary": 0.25,
    "keyword": 0.25,
    "semantic": 0.5,
}

# ── 关键词提取配置 ───────────────────────────────────────────────
# 摘要文本的最大字符数，超过则截断。
MAX_SUMMARY_EMBEDDING_CHARS = 240

# 从文本中提取的最大关键词数量。
MAX_KEYWORD_TERMS = 16

# 中文关键词停用词表：这些高频虚词对检索无区分度，提取关键词时过滤掉。
# 模块级线程池，复用于 sync→async 委托，避免重复创建/销毁线程
_sync_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

KEYWORD_STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "或",
    "是",
    "在",
    "对",
    "及",
    "以及",
    "等",
    "为",
    "有",
    "可以",
    "需要",
    "如果",
    "一个",
    "这个",
    "那个",
    "进行",
    "使用",
    "用户",
    "商品",
}


class RAGService:
    """轻量 RAG 服务 —— 本地知识库检索引擎。

    核心能力：
    1. 可配置 embedding provider + FAISS 多向量检索（摘要/关键词/语义）
    2. BM25 关键词检索（基于 jieba 中文分词）
    3. 向量 + BM25 混合召回（加权融合排序）
    4. 可配置 reranker provider 精排（CrossEncoder 重排序）
    5. 文档增删改同步（增量向量索引扩展 + 批量重建）
    6. 多层缓存（嵌入/检索/重排序三级缓存，降低 API 调用成本）

    数据持久化：
    - data/faiss.index   —— FAISS 向量索引文件
    - data/metadata.json —— 所有 chunk 记录（含嵌入文本、元数据）
    - data/documents.json —— 文档清单（含哈希、版本号、分块配置）
    """

    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
        embedding_model_name: str | None = None,
        embedding_model_kwargs: dict[str, Any] | None = None,
        embeddings: Any | None = None,
        reranker_provider: str = DEFAULT_RERANKER_PROVIDER,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
        reranker_model_kwargs: dict[str, Any] | None = None,
        reranker: Any | None = None,
        reranker_device: str | None = None,
        reranker_batch_size: int = 16,
        default_use_bm25: bool = True,
        default_use_rerank: bool = False,
        default_candidate_top_k: int = 20,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        parent_chunk_size: int | None = None,
        parent_chunk_overlap: int | None = None,
        trace_recorder: TraceRecorder | None = None,
        multi_vector_weights: dict[str, float] | None = None,
        cache: JsonCache | None = None,
        cache_ttls: dict[str, Any] | None = None,
    ) -> None:
        """初始化 RAG 服务，加载或创建向量索引和文档元数据。

        Args:
            data_dir: 数据持久化目录，存放 FAISS 索引和元数据 JSON 文件
            model_name: 嵌入模型名称（向后兼容，会被 embedding_model_name 覆盖）
            embedding_provider: 嵌入服务提供商（如 dashscope）
            embedding_model_name: 嵌入模型名称
            embedding_model_kwargs: 嵌入模型额外参数
            embeddings: 外部嵌入模型实例，不传则通过工厂创建
            reranker_provider: 重排序服务提供商
            reranker_model_name: 重排序模型名称
            reranker_model_kwargs: 重排序模型额外参数
            reranker: 外部重排序模型实例，不传则懒加载
            reranker_device: 重排序模型运行设备（cpu/cuda）
            reranker_batch_size: 重排序时的批处理大小
            default_use_bm25: 是否默认启用 BM25 检索
            default_use_rerank: 是否默认启用 CrossEncoder 重排序
            default_candidate_top_k: 候选召回数量上限
            chunk_size: 子分块大小（字符数）
            chunk_overlap: 子分块之间的重叠字符数
            parent_chunk_size: 父分块大小，默认 chunk_size * 3
            parent_chunk_overlap: 父分块之间的重叠字符数
            trace_recorder: 可选的追踪记录器，用于记录检索/排序事件
            multi_vector_weights: 多向量融合权重字典
            cache: JSON 文件缓存实例
            cache_ttls: 缓存过期时间配置字典
        """
        # ── 数据目录与持久化路径 ──
        # 所有索引和元数据都保存在 data_dir，方便持久化复用。
        self.base_dir = Path(data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 三个核心持久化文件：
        # faiss.index   → FAISS 向量索引
        # metadata.json → 所有 chunk 记录（内容、嵌入文本、元数据）
        # documents.json→ 文档清单（源文件路径、哈希、分块参数、版本）
        self.index_path = self.base_dir / "faiss.index"
        self.metadata_path = self.base_dir / "metadata.json"
        self.documents_path = self.base_dir / "documents.json"

        # ── 分块参数配置 ──
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 父块大小默认为子块的 3 倍，确保父块能覆盖其所有子块的内容
        self.parent_chunk_size = parent_chunk_size or max(chunk_size * 3, chunk_size)
        # 父块重叠默认为子块重叠的 2 倍，但不能超过父块大小
        self.parent_chunk_overlap = (
            min(chunk_overlap * 2, self.parent_chunk_size - 1) if parent_chunk_overlap is None else parent_chunk_overlap
        )
        # 校验分块参数合法性（重叠必须小于大小，子块不能大于父块等）
        self._validate_chunking_config(
            child_chunk_size=self.chunk_size,
            child_chunk_overlap=self.chunk_overlap,
            parent_chunk_size=self.parent_chunk_size,
            parent_chunk_overlap=self.parent_chunk_overlap,
        )

        # ── 嵌入模型配置 ──
        # 向量模型通过 provider 工厂创建，也支持外部传入自定义 embeddings。
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name or model_name
        self.embedding_model_kwargs = dict(embedding_model_kwargs or {})
        # 根据模型配置生成指纹哈希，用于检测模型是否变更（从而触发索引重建）
        self.embedding_config_hash = model_config_fingerprint(
            self.embedding_provider,
            self.embedding_model_name,
            self.embedding_model_kwargs,
        )
        # 创建嵌入模型实例：优先使用外部传入的，否则通过工厂函数创建
        self.embeddings = embeddings or create_embeddings(
            provider=self.embedding_provider,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs,
        )

        # ── 重排序模型配置 ──
        # 重排序模型按 provider 懒加载，避免只做入库时也初始化 reranker。
        # reranker 默认为 None，首次调用 rerank() 时通过 _get_reranker() 懒加载
        self.reranker_provider = reranker_provider
        self.reranker_model_name = reranker_model_name
        self.reranker_model_kwargs = dict(reranker_model_kwargs or {})
        self.reranker = reranker
        self.reranker_device = reranker_device
        self.reranker_batch_size = reranker_batch_size

        # ── 检索行为默认值 ──
        self.default_use_bm25 = default_use_bm25
        self.default_use_rerank = default_use_rerank
        self.default_candidate_top_k = default_candidate_top_k

        # ── 追踪与缓存 ──
        self.trace_recorder = trace_recorder
        # 多向量融合权重：归一化处理用户传入的权重
        self.multi_vector_weights = self._normalize_multi_vector_weights(multi_vector_weights)
        self.cache = cache
        self.cache_ttls = CacheTTLs.from_mapping(cache_ttls)
        # 缓存版本标识，在知识库内容变更时自动失效
        self._cache_index_version: str | None = None

        # ── 初始化数据：加载文档清单和 chunk 记录 ──
        # 加载 documents.json 文档清单
        self.documents = self._load_documents_manifest()
        # 加载 metadata.json 中的所有 chunk 记录
        self.records = self._load_records()

        # 检查是否有新 chunk 需要补充到文档清单中（兼容旧数据迁移）
        documents_changed = self._reconcile_documents_manifest()
        # 刷新嵌入配置元数据（provider、model、config_hash），检测变更
        embedding_metadata_changed = self._refresh_embedding_metadata()

        # ── 构建索引 ──
        self.bm25: BM25Plus | None = None
        # 构建多向量行列表：每个 record 拆成 3 个向量行（summary/keyword/semantic）
        self.vector_rows = self._build_vector_rows()
        # 加载或创建 FAISS 向量索引：若嵌入配置变更则强制重建
        self.index = self._load_or_create_index(force_rebuild=embedding_metadata_changed)
        # 重建 BM25 索引（基于当前 chunk 内容的 jieba 分词）
        self._rebuild_bm25()

        # 元数据有变更则持久化
        if documents_changed or embedding_metadata_changed:
            self._persist()

    def add_documents(
        self,
        file_paths: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        parent_chunk_size: int | None = None,
        parent_chunk_overlap: int | None = None,
    ) -> dict:
        """添加或更新文档（upsert_documents 的别名）。

        只有内容哈希或分块配置发生变化的文档才会被重新索引。
        该方法是幂等的：多次传入相同文件不会产生重复 chunk。

        Returns:
            包含 added_chunks/add_documents/updated_documents/skipped_documents 等字段的 dict
        """
        return self.upsert_documents(
            file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
        )

    def upsert_documents(
        self,
        file_paths: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        parent_chunk_size: int | None = None,
        parent_chunk_overlap: int | None = None,
    ) -> dict:
        """添加新文档或更新已变更的文档到本地索引（幂等操作）。

        核心流程：
        1. 遍历每个文件，计算文档级哈希和 chunk 级哈希
        2. 与已有文档的 source_hash + 分块配置对比，判断是否变更
        3. 对变更的文档：先删除旧 chunk，再添加新 chunk
        4. 决策索引更新策略：有删除则全量重建，仅新增则增量扩展
        5. 持久化 BM25、FAISS 索引和元数据

        幂等保证：同一文件多次 upsert 不会重复创建 chunk，因为会先检查哈希是否变更。
        """
        # 使用传入参数或默认配置的分块参数
        actual_chunk_size = chunk_size or self.chunk_size
        actual_chunk_overlap = self.chunk_overlap if chunk_overlap is None else chunk_overlap
        actual_parent_chunk_size = parent_chunk_size or self.parent_chunk_size
        actual_parent_chunk_overlap = (
            self.parent_chunk_overlap if parent_chunk_overlap is None else parent_chunk_overlap
        )
        # 校验分块配置合法性
        self._validate_chunking_config(
            child_chunk_size=actual_chunk_size,
            child_chunk_overlap=actual_chunk_overlap,
            parent_chunk_size=actual_parent_chunk_size,
            parent_chunk_overlap=actual_parent_chunk_overlap,
        )

        # ── 分类统计变量 ──
        records_to_add: list[dict] = []
        added_sources: list[str] = []  # 新增文档路径
        updated_sources: list[str] = []  # 更新文档路径
        skipped_sources: list[str] = []  # 跳过的文档路径（内容未变）
        deleted_chunks = 0  # 删除的旧 chunk 数量
        added_parent_chunks = 0  # 新增的父块数量
        changed_documents = False  # 是否有任何文档发生了变更

        # ── 遍历每个文件，按需索引 ──
        for file_path in file_paths:
            path = Path(file_path)
            # 读取文件内容、分块、计算哈希
            new_records, document_info = self._build_document_records(
                path,
                actual_chunk_size,
                actual_chunk_overlap,
                actual_parent_chunk_size,
                actual_parent_chunk_overlap,
            )
            doc_id = document_info["doc_id"]
            existing_info = self.documents.get(doc_id)

            # 检查文档内容和分块配置是否都未变化，是则跳过
            if self._document_is_unchanged(
                existing_info,
                document_info,
                actual_chunk_size,
                actual_chunk_overlap,
                actual_parent_chunk_size,
                actual_parent_chunk_overlap,
            ):
                skipped_sources.append(document_info["source"])
                continue

            # 文档有变更：先删除旧 chunk（可能 0 条，即新文档）
            existing_chunk_count = self._remove_document_records(doc_id)
            deleted_chunks += existing_chunk_count

            # 根据是否存在旧信息判断是新增还是更新，并设置版本号
            if existing_info is None:
                added_sources.append(document_info["source"])
                document_info["version"] = 1  # 新文档版本号为 1
            else:
                updated_sources.append(document_info["source"])
                document_info["version"] = int(existing_info.get("version") or 0) + 1  # 版本号递增

            # 注册文档信息并收集新的 chunk 记录
            self.documents[doc_id] = document_info
            records_to_add.extend(new_records)
            added_parent_chunks += int(document_info.get("parent_chunk_count") or 0)
            changed_documents = True

        # 将新增的 chunk 记录追加到全局记录列表
        if records_to_add:
            self.records.extend(records_to_add)

        # ── 决策索引更新方式 ──
        if changed_documents:
            if deleted_chunks > 0:
                # 有删除操作 → 全量重建 FAISS 索引（FAISS 不支持删除单条向量）
                self._rebuild_vector_index()
            elif records_to_add:
                # 仅有新增 → 增量扩展 FAISS 索引（性能更优，避免重建已有向量）
                self._extend_vector_index(records_to_add)
            # BM25 索引每次变更都需要全量重建（corpus 变了）
            self._rebuild_bm25()
            # 持久化 FAISS 索引和元数据到磁盘
            self._persist()

        # ── 组装结果 ──
        result = {
            "added_chunks": len(records_to_add),
            "added_parent_chunks": added_parent_chunks,
            "deleted_chunks": deleted_chunks,
            "sources": sorted({record["source"] for record in records_to_add}),
            "added_documents": sorted(added_sources),
            "updated_documents": sorted(updated_sources),
            "skipped_documents": sorted(skipped_sources),
        }
        # 记录追踪事件
        self._trace_event("rag.upsert_documents", result)
        return result

    def update_document(
        self,
        file_path: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        parent_chunk_size: int | None = None,
        parent_chunk_overlap: int | None = None,
    ) -> dict:
        """更新单个已索引的文档，仅当文件内容变更时才重新索引。

        实际调用 upsert_documents([file_path])，共享相同的幂等逻辑。
        """
        return self.upsert_documents(
            [file_path],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
        )

    def delete_document(self, document_ref: str) -> dict:
        """删除指定文档及其所有 chunk，并重建索引。

        支持两种删除方式：
        - 通过 doc_id（SHA256 哈希）直接删除
        - 通过源文件路径（source path）模糊匹配后删除

        删除后必须全量重建 FAISS 索引（FAISS 不支持删除单条向量）。
        """
        # 尝试将引用解析为 doc_id（支持路径→doc_id 的转换）
        doc_id = self._resolve_document_id(document_ref)
        if doc_id is None:
            # 文档不存在，返回空结果
            result = {
                "deleted_chunks": 0,
                "document_id": None,
                "source": document_ref,
            }
            self._trace_event("rag.delete_document", result)
            return result

        # 从文档清单中移除并获取文档信息
        document_info = self.documents.pop(doc_id, {})
        # 从 records 列表中过滤掉该文档的所有 chunk
        deleted_chunks = self._remove_document_records(doc_id)
        # 全量重建向量索引和 BM25 索引
        self._rebuild_vector_index()
        self._rebuild_bm25()
        # 持久化到磁盘
        self._persist()
        result = {
            "deleted_chunks": deleted_chunks,
            "document_id": doc_id,
            "source": document_info.get("source", document_ref),
        }
        self._trace_event("rag.delete_document", result)
        return result

    def sync_documents(
        self,
        file_paths: list[str],
        *,
        remove_missing: bool = False,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        parent_chunk_size: int | None = None,
        parent_chunk_overlap: int | None = None,
    ) -> dict:
        """同步文档：upsert 指定的文件，并可选择性地删除不在列表中的文档。

        这是批量文档管理的首选方法：
        - 传入当前所有文件的列表 → upsert 新增/更新的文件
        - remove_missing=True → 删除已索引但不在文件列表中的文档（类似 rsync 行为）

        注意：当 remove_missing=True 且有文档被删除时，会全量重建 FAISS 索引。
        """
        # 先执行 upsert 处理传入的文件
        result = self.upsert_documents(
            file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
        )
        # 不删除多余文档则直接返回
        if not remove_missing:
            return result

        # 计算期望保留的文档 ID 集合
        desired_doc_ids = {self._document_id_for_path(Path(file_path)) for file_path in file_paths}
        # 找出不在文件列表中的多余文档并删除
        removed_sources: list[str] = []
        removed_chunks = 0
        for doc_id in sorted(set(self.documents) - desired_doc_ids):
            document_info = self.documents.pop(doc_id, {})
            removed_sources.append(str(document_info.get("source") or doc_id))
            removed_chunks += self._remove_document_records(doc_id)

        # 有删除操作则重建索引并持久化
        if removed_sources:
            self._rebuild_vector_index()
            self._rebuild_bm25()
            self._persist()

        result["removed_documents"] = removed_sources
        result["deleted_chunks"] += removed_chunks
        return result

    def list_documents(self) -> list[dict]:
        """返回当前已索引文档的元数据列表，按源文件路径排序。"""
        return sorted(
            [dict(item) for item in self.documents.values()],
            key=lambda item: str(item.get("source", "")),
        )

    # ── 追踪与缓存辅助方法 ───────────────────────────────────────

    def _trace_event(self, name: str, payload: dict[str, Any]) -> None:
        """向追踪记录器发送事件（如果配置了 trace_recorder）。

        所有检索、入库存、排序操作都会通过此方法记录追踪事件，
        事件以 JSONL 格式追加到 traces/ 目录下。
        """
        if self.trace_recorder is None:
            return
        self.trace_recorder.event("rag", name, payload)

    def _trace_search_event(
        self,
        method: str,
        start: float,
        result_count: int,
        results: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> None:
        """Emit a trace event for a search operation (skips work when tracing disabled)."""
        if self.trace_recorder is None:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000
        payload: dict[str, Any] = {"elapsed_ms": elapsed_ms, "result_count": result_count, **extra}
        if results is not None:
            payload["results"] = [summarize_result(item, include_content=True) for item in results]
        self._trace_event(f"rag.{method}", payload)

    def _knowledge_cache_version(self) -> str:
        """生成知识库缓存版本标识。

        基于文档清单、嵌入配置哈希、多向量权重和所有 chunk 的关键字段
        生成稳定的摘要哈希。当知识库内容或配置变更时，该版本号会变化，
        从而使所有旧缓存自动失效。

        Returns:
            知识库状态的稳定摘要哈希字符串
        """
        if self._cache_index_version is None:
            # 使用稳定的摘要函数生成版本标识
            # 只取 chunk 的关键字段，避免内容字符串变化导致版本不必要的漂移
            self._cache_index_version = stable_cache_digest(
                {
                    "documents": self.documents,
                    "embedding_config_hash": self.embedding_config_hash,
                    "multi_vector_weights": self.multi_vector_weights,
                    "record_count": len(self.records),
                    "records": [
                        {
                            "id": record.get("id"),
                            "doc_id": record.get("doc_id"),
                            "content_hash": record.get("content_hash"),
                            "parent_id": record.get("parent_id"),
                            "parent_content_hash": record.get("parent_content_hash"),
                        }
                        for record in self.records
                    ],
                }
            )
        return self._cache_index_version

    def _invalidate_cache_version(self) -> None:
        """使缓存版本失效，下次调用 _knowledge_cache_version() 将重新计算。

        在每次 _persist() 持久化后调用，确保缓存版本反映最新的知识库状态。
        """
        self._cache_index_version = None

    def _retrieval_cache_payload(
        self,
        method: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """构建检索缓存的有效载荷。

        将检索方法名、知识库版本、嵌入配置等信息与调用参数合并，
        生成一个统一的缓存载荷字典。不同配置和知识库版本的检索请求
        会产生不同的缓存键，避免返回过期结果。
        """
        return {
            "method": method,
            "knowledge_version": self._knowledge_cache_version(),
            "embedding_provider": self.embedding_provider,
            "embedding_model_name": self.embedding_model_name,
            "embedding_config_hash": self.embedding_config_hash,
            **kwargs,  # 合并调用方传入的额外参数（如 query、top_k、vector_weight 等）
        }

    def search_by_vector(self, query: str, top_k: int = 3) -> list[dict]:
        """纯多向量检索（不使用 BM25 和重排序）。

        流程：
        1. 将查询文本嵌入为查询向量（L2 归一化）
        2. 在 FAISS 索引中搜索最相似的向量行
        3. 按多向量权重聚合同一 chunk 的 summary/keyword/semantic 三种分数
        4. 按父块去重（一个父块的多个子块命中时只保留最高分）
        5. 返回 top_k 结果
        """
        start = time.perf_counter()
        # 空查询或无数据时直接返回空列表
        if not query.strip() or not self.records or self.index.ntotal == 0:
            self._trace_event(
                "rag.search_by_vector",
                {
                    "query": query,
                    "top_k": top_k,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
            return []

        # 计算候选召回数量上限：top_k × OVERFETCH_FACTOR，确保去重后有足够结果
        limit = self._candidate_limit(top_k)
        # 获取向量检索的详细信息（每个 chunk 的三类向量分数和匹配情况）
        vector_details = self._vector_score_details_map(query, limit)

        # 将向量分数详情转换为统一的结果格式
        candidate_results = []
        for idx, details in vector_details.items():
            vector_score = float(details["score"])
            candidate_results.append(
                self._build_result(
                    idx=idx,
                    score=vector_score,
                    vector_score=vector_score,
                    bm25_score=0.0,  # 纯向量模式，BM25 分数为 0
                    hybrid_score=vector_score,  # 纯向量模式，混合分数等于向量分数
                    rerank_score=None,
                    retrieval_mode="multi_vector",
                    multi_vector_scores=details["scores"],
                    matched_vector_types=details["matched_vector_types"],
                    best_vector_type=details["best_vector_type"],
                )
            )

        # 按分数降序排列，去重后截取 top_k
        candidate_results.sort(key=lambda item: item["score"], reverse=True)
        results = self._deduplicate_parent_results(candidate_results)[:top_k]

        # 记录追踪事件
        self._trace_event(
            "rag.search_by_vector",
            {
                "query": query,
                "top_k": top_k,
                "candidate_count": len(candidate_results),
                "vector_row_count": len(self.vector_rows),
                "result_count": len(results),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "results": [summarize_result(item, include_content=True) for item in results],
            },
        )
        return results

    def search_by_bm25(
        self,
        query: str,
        top_k: int = 3,
        use_bm25: bool | None = None,
    ) -> list[dict]:
        """纯 BM25 关键词检索（不使用向量召回）。

        流程：
        1. 用 jieba 对查询文本分词
        2. 调用 BM25Plus 计算每个 chunk 与查询的相关性分数
        3. 分数归一化（除以最大分数，映射到 [0, 1]）
        4. 按父块去重后返回 top_k 结果

        BM25 擅长精确关键词匹配，与向量检索的语义匹配互补。
        """
        start = time.perf_counter()
        # 确定是否启用 BM25（优先用传入参数，否则用默认配置）
        actual_use_bm25 = self.default_use_bm25 if use_bm25 is None else use_bm25
        if not actual_use_bm25:
            self._trace_event(
                "rag.search_by_bm25",
                {
                    "query": query,
                    "top_k": top_k,
                    "use_bm25": False,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
            return []

        # 空查询或无数据时返回空
        if not query.strip() or not self.records or self.bm25 is None:
            self._trace_event(
                "rag.search_by_bm25",
                {
                    "query": query,
                    "top_k": top_k,
                    "use_bm25": actual_use_bm25,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
            return []

        # jieba 分词：将查询转为 BM25 可以处理的 token 列表
        query_tokens = self._tokenize(query)
        if not query_tokens:
            self._trace_event(
                "rag.search_by_bm25",
                {
                    "query": query,
                    "top_k": top_k,
                    "use_bm25": actual_use_bm25,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
            return []

        # BM25Plus 计算每个文档的原始分数
        raw_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype="float32")
        # 所有分数都 <= 0 表示没有匹配，返回空
        if raw_scores.size == 0 or float(raw_scores.max()) <= 0:
            self._trace_event(
                "rag.search_by_bm25",
                {
                    "query": query,
                    "top_k": top_k,
                    "use_bm25": actual_use_bm25,
                    "tokens": query_tokens,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
            return []

        # 分数归一化到 [0, 1]，方便与向量分数融合
        normalized_scores = self._normalize_bm25_scores(raw_scores)
        # 取分数最高的前 N 个候选（N = top_k × OVERFETCH_FACTOR）
        top_indices = np.argsort(raw_scores)[::-1][: self._candidate_limit(top_k)]

        # 构建候选结果列表
        candidate_results = []
        for idx in top_indices:
            if raw_scores[idx] <= 0:
                continue
            bm25_score = float(normalized_scores[idx])
            candidate_results.append(
                self._build_result(
                    idx=int(idx),
                    score=bm25_score,
                    vector_score=0.0,  # 纯 BM25 模式，向量分数为 0
                    bm25_score=bm25_score,
                    hybrid_score=bm25_score,  # 纯 BM25 模式，混合分数等于 BM25 分数
                    rerank_score=None,
                    retrieval_mode="bm25",
                )
            )

        # 按分数降序排列，去重后截取 top_k
        candidate_results.sort(key=lambda item: item["score"], reverse=True)
        results = self._deduplicate_parent_results(candidate_results)[:top_k]
        self._trace_event(
            "rag.search_by_bm25",
            {
                "query": query,
                "top_k": top_k,
                "use_bm25": actual_use_bm25,
                "tokens": query_tokens,
                "candidate_count": len(candidate_results),
                "result_count": len(results),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "results": [summarize_result(item, include_content=True) for item in results],
            },
        )
        return results

    def _search_by_hybrid_fuse(
        self,
        query: str,
        top_k: int,
        actual_vector_weight: float,
        actual_bm25_weight: float,
        actual_use_bm25: bool,
        vector_details: dict[int, Any],
        bm25_scores: dict[int, float],
        cache_payload: dict[str, Any],
        start: float,
    ) -> list[dict]:
        """加权融合、排序、去重、缓存和追踪（sync/async 共用核心逻辑）。

        融合公式：hybrid_score = vector_weight × vector_score + bm25_weight × bm25_score
        """
        vector_scores = {idx: float(details["score"]) for idx, details in vector_details.items()}
        candidate_ids = set(vector_scores) | set(bm25_scores)

        ranked_results = []
        for idx in candidate_ids:
            vs = vector_scores.get(idx, 0.0)
            bs = bm25_scores.get(idx, 0.0)
            details = vector_details.get(idx, {"scores": {}, "matched_vector_types": [], "best_vector_type": None})
            hybrid_score = actual_vector_weight * vs + actual_bm25_weight * bs
            ranked_results.append(
                self._build_result(
                    idx=idx,
                    score=hybrid_score,
                    vector_score=vs,
                    bm25_score=bs,
                    hybrid_score=hybrid_score,
                    rerank_score=None,
                    retrieval_mode="hybrid" if actual_use_bm25 else "multi_vector",
                    multi_vector_scores=details["scores"],
                    matched_vector_types=details["matched_vector_types"],
                    best_vector_type=details["best_vector_type"],
                )
            )

        ranked_results.sort(key=lambda item: item["score"], reverse=True)
        results = self._deduplicate_parent_results(ranked_results)[:top_k]

        write_cached_list(self.cache,"retrieval", cache_payload, results, ttl_s=self.cache_ttls.retrieval_ttl_s)
        self._trace_event(
            "rag.search_by_hybrid",
            {
                "query": query,
                "top_k": top_k,
                "vector_weight": actual_vector_weight,
                "bm25_weight": actual_bm25_weight,
                "use_bm25": actual_use_bm25,
                "candidate_count": len(candidate_ids),
                "vector_row_count": len(self.vector_rows),
                "result_count": len(results),
                "cache_hit": False,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "results": [summarize_result(item, include_content=True) for item in results],
            },
        )
        return results

    def search_by_hybrid(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_bm25: bool | None = None,
    ) -> list[dict]:
        """混合检索：向量召回和 BM25 召回加权融合后的结果，不做精排。"""
        coro = self.asearch_by_hybrid(query, top_k, vector_weight, bm25_weight, use_bm25)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return _sync_pool.submit(asyncio.run, coro).result()

    async def asearch_by_hybrid(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_bm25: bool | None = None,
    ) -> list[dict]:
        """异步混合检索：向量检索和 BM25 检索通过 asyncio.gather 并发执行。"""
        start = time.perf_counter()
        if not query.strip() or not self.records:
            self._trace_event(
                "rag.search_by_hybrid",
                {"query": query, "top_k": top_k, "result_count": 0, "elapsed_ms": (time.perf_counter() - start) * 1000},
            )
            return []

        actual_use_bm25 = self.default_use_bm25 if use_bm25 is None else use_bm25
        actual_vector_weight = 1.0 if not actual_use_bm25 else vector_weight
        actual_bm25_weight = bm25_weight if actual_use_bm25 else 0.0
        self._validate_weights(actual_vector_weight, actual_bm25_weight)

        cache_payload = self._retrieval_cache_payload(
            "search_by_hybrid",
            query=query,
            top_k=top_k,
            vector_weight=actual_vector_weight,
            bm25_weight=actual_bm25_weight,
            use_bm25=actual_use_bm25,
        )
        cached_results = read_cached_list(self.cache,"retrieval", cache_payload)
        if cached_results is not None:
            self._trace_event(
                "rag.search_by_hybrid",
                {
                    "query": query,
                    "top_k": top_k,
                    "vector_weight": actual_vector_weight,
                    "bm25_weight": actual_bm25_weight,
                    "use_bm25": actual_use_bm25,
                    "candidate_count": None,
                    "vector_row_count": len(self.vector_rows),
                    "result_count": len(cached_results),
                    "cache_hit": True,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                    "results": [summarize_result(item, include_content=True) for item in cached_results],
                },
            )
            return cached_results

        limit = self._candidate_limit(top_k)
        vector_task = asyncio.to_thread(self._vector_score_details_map, query, limit)
        bm25_task = asyncio.to_thread(self._bm25_score_map, query, limit) if actual_use_bm25 else None
        bm25_scores: dict[int, float] = {}
        if bm25_task is not None:
            vector_details, bm25_scores = await asyncio.gather(vector_task, bm25_task)
        else:
            vector_details = await vector_task

        return self._search_by_hybrid_fuse(
            query,
            top_k,
            actual_vector_weight,
            actual_bm25_weight,
            actual_use_bm25,
            vector_details,
            bm25_scores,
            cache_payload,
            start,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """使用 Cross-Encoder 对候选结果做精排（重排序）。

        与向量/BM25 的双编码器方式不同，Cross-Encoder 将查询和文档拼接后
        联合编码，能更准确地捕捉查询-文档的细粒度相关性，但计算开销更大。
        因此通常只对混合召回返回的候选集（如 top-20）做精排。

        流程：
        1. 检查重排序缓存
        2. 构建 (query, content) 对列表
        3. 调用 CrossEncoder 模型预测每对的相关性分数
        4. 将重排序分数合并到结果中，按新分数排序
        5. 截取 top_k 结果
        """
        start = time.perf_counter()
        # 空查询或无候选时直接返回空列表
        if not query.strip() or not candidates:
            self._trace_event(
                "rag.rerank",
                {
                    "query": query,
                    "candidate_count": len(candidates),
                    "top_k": top_k,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
            return []

        # ── 检查重排序缓存 ──
        cache_payload = self._retrieval_cache_payload(
            "rerank",
            query=query,
            candidates=candidates,
            top_k=top_k,
            reranker_provider=self.reranker_provider,
            reranker_model_name=self.reranker_model_name,
            reranker_model_kwargs=self.reranker_model_kwargs,
            reranker_device=self.reranker_device,
        )
        cached_results = read_cached_list(self.cache,"rerank", cache_payload)
        if cached_results is not None:
            self._trace_event(
                "rag.rerank",
                {
                    "query": query,
                    "reranker_provider": self.reranker_provider,
                    "reranker_model_name": self.reranker_model_name,
                    "candidate_count": len(candidates),
                    "top_k": top_k,
                    "result_count": len(cached_results),
                    "cache_hit": True,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                    "results": [summarize_result(item, include_content=True) for item in cached_results],
                },
            )
            return cached_results

        # ── 执行 CrossEncoder 预测 ──
        # 懒加载重排序模型（首次调用才初始化，节省不重排时的内存开销）
        reranker = self._get_reranker()
        # 构建 (查询, 文档内容) 配对列表
        pairs = [(query, item["content"]) for item in candidates]
        # 批量预测相关性分数
        raw_scores = reranker.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,  # 禁用进度条，减少终端噪音
        )
        # 将分数统一转换为 float 列表（处理可能的标量/矩阵等输出格式）
        rerank_scores = self._coerce_rerank_scores(raw_scores)

        # ── 合并重排序分数到结果中 ──
        reranked = []
        for item, rerank_score in zip(candidates, rerank_scores, strict=True):
            merged = dict(item)
            # 用重排序分数替换原分数（原混合分数仍保留在 hybrid_score 字段）
            merged["score"] = float(rerank_score)
            merged["rerank_score"] = float(rerank_score)
            merged["retrieval_mode"] = "hybrid_rerank"
            reranked.append(merged)

        # 按重排序分数降序排列，截取 top_k
        reranked.sort(key=lambda item: item["score"], reverse=True)
        results = reranked if top_k is None else reranked[:top_k]

        # 写入重排序缓存
        write_cached_list(self.cache,
            "rerank",
            cache_payload,
            results,
            ttl_s=self.cache_ttls.rerank_ttl_s,
        )
        # 记录追踪事件，包含候选和最终结果的摘要
        self._trace_event(
            "rag.rerank",
            {
                "query": query,
                "reranker_provider": self.reranker_provider,
                "reranker_model_name": self.reranker_model_name,
                "candidate_count": len(candidates),
                "top_k": top_k,
                "result_count": len(results),
                "cache_hit": False,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "candidates": [summarize_result(item, include_content=True) for item in candidates],
                "results": [summarize_result(item, include_content=True) for item in results],
            },
        )
        return results

    async def arerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """异步重排序：在线程池中执行同步的 rerank 方法。

        这样重排序计算不会阻塞异步事件循环，适合在 async agent 中使用。
        """
        return await asyncio.to_thread(
            self.rerank,
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_bm25: bool | None = None,
        use_rerank: bool | None = None,
        candidate_top_k: int | None = None,
    ) -> list[dict]:
        """默认搜索入口 —— 完整的 RAG 检索管线（委托异步版本执行）。"""
        coro = self.asearch(
            query, top_k, vector_weight, bm25_weight,
            use_bm25, use_rerank, candidate_top_k,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return _sync_pool.submit(asyncio.run, coro).result()

    async def asearch(
        self,
        query: str,
        top_k: int = 3,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_bm25: bool | None = None,
        use_rerank: bool | None = None,
        candidate_top_k: int | None = None,
    ) -> list[dict]:
        """异步搜索入口 —— 与 search() 逻辑一致，但使用异步混合召回和重排序。

        在异步 agent 中使用此方法可以避免阻塞事件循环，
        向量检索和 BM25 检索会并发执行以减少总延迟。
        """
        start = time.perf_counter()
        # 空查询或无数据时返回空
        if not query.strip() or not self.records:
            self._trace_event(
                "rag.search",
                {
                    "query": query,
                    "top_k": top_k,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
            return []

        # 确定重排序和候选数量配置
        actual_use_rerank = self.default_use_rerank if use_rerank is None else use_rerank
        actual_candidate_top_k = candidate_top_k or self.default_candidate_top_k
        actual_candidate_top_k = max(actual_candidate_top_k, top_k)
        actual_use_bm25 = self.default_use_bm25 if use_bm25 is None else use_bm25

        # ── 检查缓存 ──
        cache_payload = self._retrieval_cache_payload(
            "search",
            query=query,
            top_k=top_k,
            candidate_top_k=actual_candidate_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            use_bm25=actual_use_bm25,
            use_rerank=actual_use_rerank,
        )
        cached_results = read_cached_list(self.cache,"retrieval", cache_payload)
        if cached_results is not None:
            self._trace_event(
                "rag.search",
                {
                    "query": query,
                    "top_k": top_k,
                    "candidate_top_k": actual_candidate_top_k,
                    "use_bm25": actual_use_bm25,
                    "use_rerank": actual_use_rerank,
                    "candidate_count": None,
                    "result_count": len(cached_results),
                    "cache_hit": True,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                    "results": [summarize_result(item, include_content=True) for item in cached_results],
                },
            )
            return cached_results

        # ── 第一阶段：异步混合召回（向量 + BM25 并发执行）──
        hybrid_candidates = await self.asearch_by_hybrid(
            query=query,
            top_k=actual_candidate_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            use_bm25=use_bm25,
        )

        # ── 第二阶段：可选异步重排序 ──
        if not actual_use_rerank:
            results = hybrid_candidates[:top_k]
        else:
            results = await self.arerank(
                query=query,
                candidates=hybrid_candidates,
                top_k=top_k,
            )

        # 写入缓存并记录追踪
        write_cached_list(self.cache,
            "retrieval",
            cache_payload,
            results,
            ttl_s=self.cache_ttls.retrieval_ttl_s,
        )
        self._trace_event(
            "rag.search",
            {
                "query": query,
                "top_k": top_k,
                "candidate_top_k": actual_candidate_top_k,
                "use_bm25": actual_use_bm25,
                "use_rerank": actual_use_rerank,
                "candidate_count": len(hybrid_candidates),
                "result_count": len(results),
                "cache_hit": False,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "results": [summarize_result(item, include_content=True) for item in results],
            },
        )
        return results

    def reset(self) -> None:
        """清空所有索引数据和元数据，恢复到初始空状态。

        操作包括：清空 records 列表、documents 字典、vector_rows 列表，
        重建空的 FAISS 索引和 BM25 索引，并持久化到磁盘。
        用于完全重置知识库而不需要删除 data/ 目录。
        """
        self.records = []
        self.documents = {}
        self.vector_rows = []
        # 创建一个空的 FAISS 索引（维度为 0，添加向量时会自动建立正确维度）
        self.index = self._create_index()
        # 重建 BM25 索引（空语料，设 self.bm25 = None）
        self._rebuild_bm25()
        # 持久化空状态到磁盘
        self._persist()

    def _build_document_records(
        self,
        path: Path,
        chunk_size: int,
        chunk_overlap: int,
        parent_chunk_size: int,
        parent_chunk_overlap: int,
    ) -> tuple[list[dict], dict]:
        """读取文件并构建父子分块记录。

        父子分块策略：
        - 先将全文按 parent_chunk_size 切分为父块（大粒度，用于展示上下文）
        - 再将每个父块按 chunk_size 切分为子块（小粒度，用于精确检索）
        - 每个子块保留其所属父块的引用（parent_id）

        检索时返回父块内容（更完整），索引时使用子块的内容哈希和嵌入文本。

        Returns:
            (records, document_info) 元组
            - records: 子块记录列表
            - document_info: 文档元信息字典
        """
        # 读取文件内容
        text = self._read_document(path)
        source = str(path)
        # 根据文件路径生成全局唯一的 doc_id（SHA256 前 24 位）
        doc_id = self._document_id_for_path(path)
        # 文件内容 SHA256 哈希，用于检测文档变更
        source_hash = self._hash_text(text)
        # 先按父块大小切分全文
        parent_chunks = self._split_text(
            text,
            parent_chunk_size,
            parent_chunk_overlap,
        )
        indexed_at = utc_now()

        # 遍历每个父块，内层再切分为子块
        records: list[dict] = []
        for parent_index, parent_content in enumerate(parent_chunks):
            parent_content_hash = self._hash_text(parent_content)
            # 父块 ID 格式：doc_id:parent:索引:哈希前12位
            parent_id = f"{doc_id}:parent:{parent_index}:{parent_content_hash[:12]}"
            # 将父块内容按子块大小再切分
            child_chunks = self._split_text(parent_content, chunk_size, chunk_overlap)
            for child_index, child_content in enumerate(child_chunks):
                records.append(
                    self._build_record(
                        doc_id=doc_id,
                        source=source,
                        source_hash=source_hash,
                        child_content=child_content,
                        child_chunk_index=len(records),  # 全局递增的子块索引
                        child_index=child_index,  # 父块内的子块序号
                        parent_id=parent_id,
                        parent_index=parent_index,
                        parent_content=parent_content,  # 保留完整父块内容
                        parent_content_hash=parent_content_hash,
                    )
                )

        # 文档元信息，保存在 documents.json 中
        document_info = {
            "doc_id": doc_id,
            "source": source,
            "source_hash": source_hash,
            "chunk_count": len(records),  # 子块总数
            "parent_chunk_count": len(parent_chunks),  # 父块总数
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "parent_chunk_size": parent_chunk_size,
            "parent_chunk_overlap": parent_chunk_overlap,
            "chunking_strategy": CHUNKING_STRATEGY,
            "embedding_strategy": MULTI_VECTOR_STRATEGY,
            "embedding_types": list(MULTI_VECTOR_TYPES),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model_name,
            "embedding_config_hash": self.embedding_config_hash,
            "indexed_at": indexed_at,
            "version": 1,
        }
        return records, document_info

    def _build_record(
        self,
        *,
        doc_id: str,
        source: str,
        source_hash: str,
        child_content: str,
        child_chunk_index: int,
        child_index: int,
        parent_id: str,
        parent_index: int,
        parent_content: str,
        parent_content_hash: str,
    ) -> dict:
        """构建单个子块的完整记录。

        每条记录包含：
        - id: 全局唯一 chunk ID（包含 doc_id、父块索引、子块索引、内容哈希）
        - content: 子块文本（用于精确检索嵌入）
        - embedding_texts: 三类嵌入文本（摘要、关键词、语义）
        - parent_content: 父块完整文本（检索返回用，提供更完整上下文）
        - metadata: 包含层级关系和配置快照的元数据字典
        """
        content_hash = self._hash_text(child_content)
        # 子块 ID：doc_id:parent:父索引:child:子索引:内容哈希前12位
        child_chunk_id = f"{doc_id}:parent:{parent_index}:child:{child_index}:{content_hash[:12]}"
        return {
            "id": child_chunk_id,
            "doc_id": doc_id,
            "source": source,
            "source_hash": source_hash,
            "content_hash": content_hash,
            "content": child_content,  # 子块原始文本
            # 生成三种嵌入文本：摘要、关键词、语义
            "embedding_texts": self._build_embedding_texts(child_content),
            "parent_id": parent_id,
            "parent_content_hash": parent_content_hash,
            "parent_content": parent_content,  # 父块完整内容（检索返回用）
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": parent_id,  # 兼容旧字段，指向父块
                "chunk_index": parent_index,  # 兼容旧字段，父块索引
                "parent_id": parent_id,
                "parent_index": parent_index,
                "parent_content_hash": parent_content_hash,
                "child_chunk_id": child_chunk_id,
                "child_chunk_index": child_chunk_index,
                "child_index": child_index,
                "source_hash": source_hash,
                "chunking_strategy": CHUNKING_STRATEGY,
                "embedding_strategy": MULTI_VECTOR_STRATEGY,
                "embedding_types": list(MULTI_VECTOR_TYPES),
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model_name,
                "embedding_config_hash": self.embedding_config_hash,
            },
        }

    def _document_is_unchanged(
        self,
        existing_info: dict | None,
        new_info: dict,
        chunk_size: int,
        chunk_overlap: int,
        parent_chunk_size: int,
        parent_chunk_overlap: int,
    ) -> bool:
        """判断文档是否需要重新索引。

        需要重新索引的条件（任一不满足即变更）：
        1. 源文件哈希（source_hash）相同 → 文件内容未变
        2. 所有分块参数相同 → 分块方式未变
        3. 分块策略和嵌入策略相同 → 索引策略未变
        4. 嵌入模型 provider + model_name + config_hash 相同 → 模型配置未变

        只有当所有条件都满足时才认为是"未变更"，可跳过重索引。
        """
        if existing_info is None:
            return False
        return (
            existing_info.get("source_hash") == new_info.get("source_hash")
            and int(existing_info.get("chunk_size") or 0) == chunk_size
            and int(existing_info.get("chunk_overlap") or 0) == chunk_overlap
            and int(existing_info.get("parent_chunk_size") or 0) == parent_chunk_size
            and int(existing_info.get("parent_chunk_overlap") or 0) == parent_chunk_overlap
            and existing_info.get("chunking_strategy") == CHUNKING_STRATEGY
            and existing_info.get("embedding_strategy") == MULTI_VECTOR_STRATEGY
            and existing_info.get("embedding_provider") == self.embedding_provider
            and existing_info.get("embedding_model") == self.embedding_model_name
            and existing_info.get("embedding_config_hash") == self.embedding_config_hash
        )

    def _remove_document_records(self, doc_id: str) -> int:
        """从 records 列表中移除指定文档的所有 chunk 记录。

        通过列表推导式过滤掉 doc_id 匹配的记录（从 record 顶层或 metadata 中查找）。
        返回被删除的记录数量。
        """
        before = len(self.records)
        self.records = [
            record
            for record in self.records
            # 兼容两种 doc_id 存储位置：顶层或 metadata 内
            if str(record.get("doc_id") or record.get("metadata", {}).get("doc_id")) != doc_id
        ]
        return before - len(self.records)

    def _resolve_document_id(self, document_ref: str) -> str | None:
        """将文档引用解析为 doc_id。

        支持的引用形式（按优先级）：
        1. 直接 doc_id 匹配（SHA256 哈希）
        2. 路径 → doc_id 转换匹配
        3. 源文件路径模糊匹配（支持相对路径解析）

        Returns:
            匹配的 doc_id，如果找不到则返回 None
        """
        # 直接匹配 doc_id
        if document_ref in self.documents:
            return document_ref

        # 通过路径转换匹配
        path_doc_id = self._document_id_for_path(Path(document_ref))
        if path_doc_id in self.documents:
            return path_doc_id

        # 路径模糊匹配：展开 ~ 和相对路径后进行绝对路径比对
        ref_path = Path(document_ref).expanduser()
        if not ref_path.is_absolute():
            # 相对路径以 RAG Server 项目根目录为基准解析
            ref_path = self.base_dir.parent / ref_path
        ref_resolved = ref_path.resolve(strict=False)
        for doc_id, document_info in self.documents.items():
            source = str(document_info.get("source") or "")
            if source == document_ref:
                return doc_id
            source_path = Path(source).expanduser()
            if not source_path.is_absolute():
                source_path = self.base_dir.parent / source_path
            source_resolved = source_path.resolve(strict=False)
            if source_resolved == ref_resolved:
                return doc_id
        return None

    def _read_document(self, path: Path) -> str:
        """读取文档文本内容。

        支持的格式：
        - .txt / .md：UTF-8 纯文本读取
        - .pdf：通过 pypdf 提取所有页面的文本
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # 纯文本文件直接读取
        if path.suffix.lower() in {".txt", ".md"}:
            return path.read_text(encoding="utf-8").strip()

        # PDF 文件逐页提取文本
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """递归字符分块：按自然语义边界将文本切分为指定大小的块。

        分块优先级：段落分隔(\\n\\n) > 换行(\\n) > 中文句号(。) > 感叹号(！)
        > 问号(？) > 分号(；) > 逗号(，) > 空格( ) > 字符级切片。
        优先按段落和中文标点切，能比纯字符切片保留更多自然语义边界。
        """
        if not text:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 分隔符按优先级从高到低排列：优先在更大的语义边界处切分
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        return splitter.split_text(text)

    def _build_embedding_texts(self, content: str) -> dict[str, str]:
        """基于子块内容生成三种嵌入文本。

        - summary（摘要）：取前 2 句 + 截断到 240 字符
        - keyword（关键词）：jieba 分词后提取高频实词（最多 16 个）
        - semantic（语义）：空白归一化后的原文
        每种嵌入文本都带有类型前缀（如 '摘要: xxx'），帮助嵌入模型理解文本用途。
        """
        normalized = self._normalize_whitespace(content)
        summary = self._chunk_summary_text(normalized)
        keywords = self._chunk_keyword_text(normalized)
        return {
            "summary": f"摘要: {summary or normalized}",
            "keyword": f"关键词: {keywords or summary or normalized}",
            "semantic": f"语义: {normalized}",
        }

    def _normalize_embedding_texts(self, value: Any, content: str) -> dict[str, str]:
        """规范化嵌入文本：如果已有嵌入文本则检验补全，否则重新生成。

        确保三种向量类型的嵌入文本都存在且非空。
        """
        defaults = self._build_embedding_texts(content)
        if not isinstance(value, dict):
            return defaults

        # 存在旧嵌入文本时逐个检查，缺失的用默认值补全
        normalized = {}
        for vector_type in MULTI_VECTOR_TYPES:
            text = str(value.get(vector_type) or "").strip()
            normalized[vector_type] = text or defaults[vector_type]
        return normalized

    def _refresh_embedding_metadata(self) -> bool:
        """刷新嵌入配置元数据，检测与上次启动相比是否发生变更。

        检查三个方面：
        1. 文档清单中的 provider/model/config_hash
        2. 每条 chunk 记录的 metadata 中的 provider/model/config_hash

        当用户切换嵌入模型时，这些元数据会发生变化，从而触发强制索引重建。

        Returns:
            是否有任何元数据发生变更
        """
        changed = False
        # 更新文档清单中的嵌入配置
        for document_info in self.documents.values():
            if document_info.get("embedding_provider") != self.embedding_provider:
                document_info["embedding_provider"] = self.embedding_provider
                changed = True
            if document_info.get("embedding_model") != self.embedding_model_name:
                document_info["embedding_model"] = self.embedding_model_name
                changed = True
            if document_info.get("embedding_config_hash") != self.embedding_config_hash:
                document_info["embedding_config_hash"] = self.embedding_config_hash
                changed = True

        # 更新每条 chunk 记录的 metadata 中的嵌入配置
        for record in self.records:
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                record["metadata"] = metadata
            for key, value in {
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model_name,
                "embedding_config_hash": self.embedding_config_hash,
            }.items():
                if metadata.get(key) != value:
                    metadata[key] = value
                    changed = True
        return changed

    def _build_vector_rows(self) -> list[dict]:
        """构建多向量行列表：将每个 chunk 记录展开为 3 个向量行。

        每个子块生成 3 个向量行（summary/keyword/semantic），
        分别对应不同的嵌入文本，检索时三者的分数按权重聚合。

        Returns:
            向量行列表，每行包含 {record_index, vector_type, text}
        """
        rows: list[dict] = []
        for record_index, record in enumerate(self.records):
            embedding_texts = self._normalize_embedding_texts(
                record.get("embedding_texts"),
                str(record.get("content") or ""),
            )
            # 回写嵌入文本到 record
            record["embedding_texts"] = embedding_texts
            # 为每种向量类型创建一行
            for vector_type in MULTI_VECTOR_TYPES:
                rows.append(
                    {
                        "record_index": record_index,  # 指向原始 chunk 的索引
                        "vector_type": vector_type,  # summary/keyword/semantic
                        "text": embedding_texts[vector_type],
                    }
                )
        return rows

    def _chunk_summary_text(self, text: str) -> str:
        """从文本中提取摘要文本（取前 2 句，截断到 240 字符）。

        按中文标点（。！？；）和换行符分割句子，取前两句作为摘要。
        适用于嵌入检索中的 'summary' 向量类型。
        """
        if not text:
            return ""

        # 按句尾标点分割句子
        sentences = [item.strip() for item in re.split(r"(?<=[。！？；.!?;])\s*|\n+", text) if item.strip()]
        # 取前两句拼接
        summary = " ".join(sentences[:2]) if sentences else text
        if len(summary) <= MAX_SUMMARY_EMBEDDING_CHARS:
            return summary
        return summary[:MAX_SUMMARY_EMBEDDING_CHARS].rstrip()

    def _chunk_keyword_text(self, text: str) -> str:
        """从文本中提取关键词并拼接为空格分隔的字符串。"""
        keywords = self._extract_keywords(text)
        return " ".join(keywords)

    def _extract_keywords(self, text: str) -> list[str]:
        """从文本中提取排名靠前的关键词（最多 MAX_KEYWORD_TERMS 个）。

        提取算法：
        1. jieba 分词 → 过滤停用词和纯符号 → 统计词频
        2. 按词频降序排列，同频的按首次出现顺序和字典序作为 tie-breaker
        3. 返回前 MAX_KEYWORD_TERMS 个关键词

        首次出现顺序的 tie-breaker 使得越早出现的词在频率相同时排名越靠前。
        """
        # first_seen: 记录每个 token 首次出现的序号，用于 tie-breaking
        first_seen: dict[str, int] = {}
        tokens: list[str] = []
        for token in self._tokenize(text):
            # 过滤停用词和无效 token
            if not self._is_keyword_token(token):
                continue
            if token not in first_seen:
                first_seen[token] = len(first_seen)  # 记录首次出现序号
            tokens.append(token)

        if not tokens:
            return []

        # 统计词频
        counts = Counter(tokens)
        # 排序规则（优先级依次递减）：
        # 1. 词频降序（-counts[token]）
        # 2. 首次出现顺序（first_seen[token]）
        # 3. 字典序（token）
        ranked = sorted(
            counts,
            key=lambda token: (-counts[token], first_seen[token], token),
        )
        return ranked[:MAX_KEYWORD_TERMS]

    def _is_keyword_token(self, token: str) -> bool:
        """判断一个分词 token 是否为有效的关键词。

        过滤规则：
        1. 在停用词表中 → 无效
        2. 只包含非字母数字字符和下划线 → 无效（纯符号）
        3. 必须包含至少一个字母/数字或中文字符 → 有效
        """
        if token in KEYWORD_STOPWORDS:
            return False
        # 正则匹配纯符号（非单词字符）
        if re.fullmatch(r"[\W_]+", token):
            return False
        # 至少包含一个字母/数字或中文字符（Unicode 范围 \u4e00-\u9fff）
        return any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in token)

    def _normalize_whitespace(self, text: str) -> str:
        """空白字符归一化：将连续的空白符（空格、换行、制表符等）替换为单个空格。"""
        return re.sub(r"\s+", " ", text).strip()

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """批量嵌入文本为向量矩阵，并进行 L2 归一化。

        L2 归一化后的向量可以通过内积（Inner Product / IndexFlatIP）计算
        余弦相似度：cos(a,b) = dot(norm(a), norm(b))。
        """
        # 调用嵌入模型批量编码
        vectors = self.embeddings.embed_documents(texts)
        matrix = np.asarray(vectors, dtype="float32")
        # L2 归一化：每行向量的模变为 1
        faiss.normalize_L2(matrix)
        return matrix

    def _embed_query(self, text: str) -> np.ndarray:
        """嵌入查询文本为查询向量（带缓存）。

        流程：
        1. 检查嵌入缓存（相同文本避免重复调用 API）
        2. 调用嵌入模型（优先 embed_query，没有则用 embed_documents）
        3. L2 归一化后返回
        4. 写入嵌入缓存

        查询嵌入缓存在知识库变更或嵌入模型配置变更时自动通过知识库版本失效。
        """
        # 构建缓存键
        cache_key = cache_key_or_none(
            self.cache,
            "embedding",
            {
                "kind": "query",
                "text": text,
                "embedding_provider": self.embedding_provider,
                "embedding_model_name": self.embedding_model_name,
                "embedding_config_hash": self.embedding_config_hash,
            },
        )
        # 检查缓存
        if cache_key is not None:
            cached = self.cache.get_json(cache_key)
            if isinstance(cached, list) and cached:
                matrix = np.asarray([cached], dtype="float32")
                if matrix.ndim == 2 and matrix.shape[1] > 0:
                    return matrix  # 缓存命中

        # 调用嵌入 API（优先用 query 专用的 embed_query 方法）
        if hasattr(self.embeddings, "embed_query"):
            vector = self.embeddings.embed_query(text)
        else:
            vector = self.embeddings.embed_documents([text])[0]
        matrix = np.asarray([vector], dtype="float32")
        # L2 归一化
        faiss.normalize_L2(matrix)

        # 写入缓存
        if cache_key is not None and self.cache is not None:
            self.cache.set_json(
                cache_key,
                matrix[0].tolist(),
                ttl_s=self.cache_ttls.embedding_ttl_s,
            )
        return matrix

    def _load_or_create_index(self, *, force_rebuild: bool = False) -> faiss.Index:
        """加载或创建 FAISS 索引。

        决策逻辑：
        1. force_rebuild → 强制重建
        2. 磁盘有索引文件且向量行数匹配 → 直接加载
        3. 索引文件行数不匹配 → 重建
        4. 有向量行但无索引文件 → 重建
        5. 无向量行 → 创建空索引
        """
        if force_rebuild:
            return self._rebuild_and_persist_vector_index()

        # 尝试从磁盘加载已有索引
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            # 检查索引中的向量数量是否与向量行数一致
            if index.ntotal == len(self.vector_rows):
                return index  # 一致，直接使用
            # 不一致（如手动修改了数据），重建
            return self._rebuild_and_persist_vector_index()

        # 有向量行但无索引文件，重建
        if self.vector_rows:
            return self._rebuild_and_persist_vector_index()

        # 完全空状态，创建空索引
        return self._create_index()

    def _create_index(self) -> faiss.Index:
        """创建空的 FAISS IndexFlatIP 索引（维度为 0）。

        IndexFlatIP = 内积索引（Inner Product），配合 L2 归一化向量，
        等价于余弦相似度检索。维度 0 意味着为空，添加第一批向量时会确定维度。
        """
        return faiss.IndexFlatIP(0)

    def _rebuild_and_persist_vector_index(self) -> faiss.Index:
        """重建向量索引并持久化到磁盘。"""
        index = self._rebuild_vector_index()
        faiss.write_index(index, str(self.index_path))
        return index

    def _rebuild_vector_index(self) -> faiss.Index:
        """全量重建 FAISS 向量索引。

        流程：
        1. 重建向量行列表（从 records 生成 3× 多向量行）
        2. 批量嵌入所有向量文本
        3. 创建新的 IndexFlatIP 并添加所有向量
        """
        # 重建向量行列表
        self.vector_rows = self._build_vector_rows()
        if not self.vector_rows:
            # 空知识库，创建空索引
            self.index = self._create_index()
            return self.index

        # 批量嵌入所有向量文本
        vectors = self._embed_texts([row["text"] for row in self.vector_rows])
        # 创建新索引并添加向量
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self.index

    def _extend_vector_index(self, new_records: list[dict]) -> None:
        """增量扩展 FAISS 索引：仅嵌入新记录并追加到已有索引。

        这是 upsert 时的性能优化路径：当只有新增文档（无删除）时，
        不需要重新嵌入所有已有向量，只需嵌入新 chunk 的向量并追加到索引末尾。

        FAISS IndexFlatIP 的 add() 方法天然支持追加，且不影响已有向量。
        """
        new_rows: list[dict] = []
        # 计算新记录在全局 records 中的起始索引
        base_record_index = len(self.records) - len(new_records)
        for offset, record in enumerate(new_records):
            record_index = base_record_index + offset
            embedding_texts = self._normalize_embedding_texts(
                record.get("embedding_texts"),
                str(record.get("content") or ""),
            )
            record["embedding_texts"] = embedding_texts
            # 为每条新记录生成 3 个向量行
            for vector_type in MULTI_VECTOR_TYPES:
                new_rows.append(
                    {
                        "record_index": record_index,
                        "vector_type": vector_type,
                        "text": embedding_texts[vector_type],
                    }
                )

        if not new_rows:
            return

        # 嵌入新向量行
        vectors = self._embed_texts([row["text"] for row in new_rows])
        # 如果当前索引维度为 0（空索引），先创建正确维度的索引
        if self.index.d == 0:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        # 追加向量到已有索引
        self.index.add(vectors)
        # 扩展向量行列表
        self.vector_rows.extend(new_rows)

    def _rebuild_bm25(self) -> None:
        """基于当前所有 chunk 的内容重建 BM25 索引。

        BM25 不单独持久化（计算量小，启动时重建即可）。
        将所有 chunk 的 jieba 分词结果作为语料，构建 BM25Plus 实例。
        小语料场景下，BM25Plus 比 BM25Okapi 更容易得到稳定的关键词分数。
        """
        # 构建分词语料：每个 chunk 内容用 jieba 分词
        tokenized_corpus = [self._tokenize(record["content"]) for record in self.records if record["content"].strip()]
        # 有语料则创建 BM25Plus 实例，否则设为 None
        self.bm25 = BM25Plus(tokenized_corpus) if tokenized_corpus else None

    def _tokenize(self, text: str) -> list[str]:
        """使用 jieba 搜索引擎模式对中文文本分词。

        lcut_for_search 相比 lcut 会额外切分复合词，提高召回率。
        例如 '清华大学' 会切分为 ['清华', '大学', '清华大学']，
        使得搜索 '大学' 或 '清华' 都能命中。

        所有 token 转为小写并去除首尾空白。
        """
        return [token.lower().strip() for token in jieba.lcut_for_search(text) if token.strip()]

    def _vector_score_details_map(self, query: str, limit: int) -> dict[int, dict]:
        """多向量检索：返回每个 chunk 的详细分数信息。

        流程：
        1. 嵌入查询向量，在 FAISS 中搜索 top row_limit 个向量行
        2. 将向量行按所属的 chunk (record_index) 分组
        3. 每个 chunk 可能有多个向量类型命中（summary/keyword/semantic），
           取每类中最高分的那个向量行
        4. 按多向量权重聚合每类的分数，得到该 chunk 的综合向量分数

        Returns:
            {record_index: {score, scores, matched_vector_types, best_vector_type, row_indices}}
        """
        # 空索引或无数据
        if self.index.ntotal == 0:
            return {}

        # 计算需要搜索的向量行数上限
        row_limit = self._vector_row_limit(limit)
        if row_limit <= 0:
            return {}

        # FAISS 内积搜索：scores[i] = 相似度分数，indices[i] = 向量行索引
        scores, indices = self.index.search(self._embed_query(query), row_limit)

        # 按 chunk (record_index) 分组收集命中信息
        hits: dict[int, dict] = {}
        for raw_score, row_idx in zip(scores[0], indices[0], strict=True):
            # 跳过无效索引（FAISS 没找到足够结果时返回 -1）
            if row_idx < 0 or row_idx >= len(self.vector_rows):
                continue

            vector_row = self.vector_rows[int(row_idx)]
            record_idx = int(vector_row["record_index"])  # 所属 chunk 的索引
            vector_type = str(vector_row["vector_type"])  # 向量类型
            # 内积分数归一化到 [0, 1]
            vector_score = normalize_vector_score(float(raw_score))

            # 初始化该 chunk 的命中记录
            hit = hits.setdefault(
                record_idx,
                {
                    "scores": {},  # {vector_type: best_score}
                    "row_indices": {},  # {vector_type: row_idx}
                },
            )

            # 对同一 chunk 的同一向量类型，只保留最高分
            previous = hit["scores"].get(vector_type)
            if previous is None or vector_score > previous:
                hit["scores"][vector_type] = vector_score
                hit["row_indices"][vector_type] = int(row_idx)

        # 组装每个 chunk 的最终结果
        result: dict[int, dict] = {}
        for record_idx, hit in hits.items():
            vector_scores = hit["scores"]
            # 按分数降序排列匹配到的向量类型
            matched_vector_types = sorted(
                vector_scores,
                key=lambda item: vector_scores[item],
                reverse=True,
            )
            result[record_idx] = {
                # 加权聚合多向量分数
                "score": self._aggregate_multi_vector_scores(vector_scores),
                # 仅保留三类标准向量类型的分数（过滤掉可能的旧类型）
                "scores": {
                    vector_type: float(vector_scores[vector_type])
                    for vector_type in MULTI_VECTOR_TYPES
                    if vector_type in vector_scores
                },
                "matched_vector_types": matched_vector_types,
                "best_vector_type": (matched_vector_types[0] if matched_vector_types else None),
                "row_indices": hit["row_indices"],
            }
        return result

    def _aggregate_multi_vector_scores(self, scores: dict[str, float]) -> float:
        """加权聚合多向量类型的分数为单一得分。

        算法：加权平均 with 缺失补偿
        - 每个向量类型的分数乘以其配置的权重，求和后除以已命中类型的权重和
        - 这样即使只有部分向量类型命中（如只匹配到语义但没匹配到摘要），
          也能得到合理的分数，不会因为权重缺失而严重偏低
        - 结果限制在 [0, 1] 范围内

        例如：配置权重 summary:0.25, keyword:0.25, semantic:0.5
        若只命中 semantic(0.8) 和 keyword(0.3)：
        score = (0.5*0.8 + 0.25*0.3) / (0.5 + 0.25) = 0.475 / 0.75 = 0.633
        """
        if not scores:
            return 0.0

        weighted_total = 0.0
        matched_weight = 0.0
        for vector_type, score in scores.items():
            weight = self.multi_vector_weights.get(vector_type, 0.0)
            if weight <= 0:
                continue
            weighted_total += weight * float(score)
            matched_weight += weight

        # 没有任何权重匹配（极端情况），退化为最大值
        if matched_weight <= 0:
            return max(float(score) for score in scores.values())
        # 加权平均，限制在 [0, 1]
        return max(0.0, min(1.0, weighted_total / matched_weight))

    def _bm25_score_map(self, query: str, limit: int) -> dict[int, float]:
        """BM25 关键词检索：返回每个 chunk 的归一化 BM25 分数映射。

        流程：
        1. jieba 分词查询文本
        2. BM25Plus 计算所有 chunk 的原始分数
        3. 分数归一化（除以最大分数）
        4. 取 top-limit 个最高分的 chunk

        Returns:
            {chunk_index: normalized_score}，分数范围 [0, 1]
        """
        if self.bm25 is None:
            return {}

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        # BM25Plus 计算原始分数
        raw_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype="float32")
        # 无匹配或所有分数 <= 0，返回空
        if raw_scores.size == 0 or float(raw_scores.max()) <= 0:
            return {}

        # 归一化到 [0, 1]
        normalized_scores = self._normalize_bm25_scores(raw_scores)
        # 按原始分数降序取前 limit 个索引
        top_indices = np.argsort(raw_scores)[::-1][:limit]
        return {
            int(idx): float(normalized_scores[idx])
            for idx in top_indices
            if raw_scores[idx] > 0  # 过滤掉分数为 0 的
        }

    def _normalize_bm25_scores(self, scores: np.ndarray) -> np.ndarray:
        """将 BM25 原始分数归一化到 [0, 1]：除以最大分数。

        BM25 分数本身无上限，归一化后方便与向量分数在相同 [0,1] 尺度上融合。
        """
        max_score = float(scores.max())
        if max_score <= 0:
            return np.zeros_like(scores)
        return scores / max_score

    def _candidate_limit(self, top_k: int) -> int:
        """计算候选召回的数量上限。

        为了防止父块去重后结果不足 top_k，先多召回一些候选：
        limit = top_k × PARENT_CHILD_OVERFETCH_FACTOR（默认 5 倍），
        但不能超过总 chunk 数量。
        """
        requested = max(1, top_k)
        return min(requested * PARENT_CHILD_OVERFETCH_FACTOR, len(self.records))

    def _vector_row_limit(self, record_limit: int) -> int:
        """计算 FAISS 向量搜索的返回行数上限。

        因为每个 chunk 对应 3 个向量行（summary/keyword/semantic），
        要在 record_limit 个 chunk 中找到 match，至少需要搜索 3× 的行数。
        """
        if not self.vector_rows:
            return 0
        requested = max(1, record_limit) * len(MULTI_VECTOR_TYPES)
        return min(requested, len(self.vector_rows))

    def _deduplicate_parent_results(self, results: list[dict]) -> list[dict]:
        """按父块去重：同一父块的多个子块命中时只保留分数最高者。

        父子分块策略中，一个父块可能被切分为多个子块。检索时多个子块
        可能同时命中，但展示时应该只返回一次父块内容。去重时保留
        第一个出现的（已经在排序中分数最高的）结果。

        去重键为 (doc_id, parent_id) 元组。
        """
        deduplicated: list[dict] = []
        seen_parent_keys: set[tuple[str, str]] = set()
        for item in results:
            parent_key = self._parent_result_key(item)
            if parent_key in seen_parent_keys:
                # 同一父块的其他子块，跳过
                continue
            seen_parent_keys.add(parent_key)
            deduplicated.append(item)
        return deduplicated

    def _parent_result_key(self, item: dict) -> tuple[str, str]:
        """从结果中提取父块去重键 (doc_id, parent_id)。

        兼容多种字段命名：parent_id、chunk_id（旧格式）、chunk_index（备用）。
        """
        metadata = item.get("metadata") or {}
        doc_id = str(item.get("doc_id") or metadata.get("doc_id") or "")
        parent_id = str(metadata.get("parent_id") or metadata.get("chunk_id") or metadata.get("chunk_index") or "")
        return doc_id, parent_id

    def _validate_weights(self, vector_weight: float, bm25_weight: float) -> None:
        """校验混合检索的权重配置合法性。

        约束：
        - 两个权重都不能为负数
        - 至少一个权重大于 0（否则所有分数都是 0，检索无意义）
        """
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("vector_weight and bm25_weight must be non-negative")
        if vector_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one retrieval weight must be greater than 0")

    def _normalize_multi_vector_weights(
        self,
        weights: dict[str, float] | None,
    ) -> dict[str, float]:
        """正则化多向量融合权重，合并默认权重。

        处理逻辑：
        1. 不传则使用默认权重
        2. 检查权重键名是否为已知的向量类型（summary/keyword/semantic）
        3. 每个权重值必须 >= 0
        4. 至少一个权重大于 0

        Returns:
            合并后的权重字典，确保所有三种类型都有值
        """
        normalized = dict(DEFAULT_MULTI_VECTOR_WEIGHTS)
        if weights is None:
            return normalized

        # 检查是否有未知的向量类型
        unknown = sorted(set(weights) - set(MULTI_VECTOR_TYPES))
        if unknown:
            raise ValueError("Unknown multi-vector weight(s): " + ", ".join(unknown))

        # 逐个覆盖默认权重
        for vector_type, raw_weight in weights.items():
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("multi-vector weights must be non-negative")
            normalized[vector_type] = weight

        # 确保至少有一个权重值 > 0
        if sum(normalized.values()) <= 0:
            raise ValueError("At least one multi-vector weight must be greater than 0")
        return normalized

    def _validate_chunking_config(
        self,
        *,
        child_chunk_size: int,
        child_chunk_overlap: int,
        parent_chunk_size: int,
        parent_chunk_overlap: int,
    ) -> None:
        """校验分块参数配置的合法性。

        约束：
        - 所有大小必须为正数
        - 重叠必须非负
        - 重叠必须小于对应的大小
        - 子块大小不能超过父块大小
        """
        if child_chunk_size <= 0 or parent_chunk_size <= 0:
            raise ValueError("chunk_size and parent_chunk_size must be positive")
        if child_chunk_overlap < 0 or parent_chunk_overlap < 0:
            raise ValueError("chunk overlaps must be non-negative")
        if child_chunk_overlap >= child_chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if parent_chunk_overlap >= parent_chunk_size:
            raise ValueError("parent_chunk_overlap must be smaller than parent_chunk_size")
        if child_chunk_size > parent_chunk_size:
            raise ValueError("chunk_size must be smaller than or equal to parent_chunk_size")

    def _get_reranker(self) -> Any:
        """获取重排序模型实例（懒加载）。

        首次调用时通过工厂函数创建 CrossEncoder 模型实例。
        之后直接返回已缓存的实例，避免重复初始化。
        懒加载的优势：不做检索/重排时不会加载大模型（约 700MB），节省内存。
        """
        if self.reranker is None:
            self.reranker = create_reranker(
                provider=self.reranker_provider,
                model_name=self.reranker_model_name,
                device=self.reranker_device,
                **self.reranker_model_kwargs,
            )
        return self.reranker

    def _coerce_rerank_scores(self, scores: Any) -> list[float]:
        """将 CrossEncoder 预测分数统一转换为 float 列表。

        不同模型可能返回不同格式：
        - 标量（单个分数）→ 转为单元素列表
        - 一维数组 → 直接转为 float 列表
        - 多维数组 → 取最后一列（某些模型可能返回多维度输出，默认最后一列为相关性分数）
        """
        array = np.asarray(scores, dtype="float32")
        if array.ndim == 0:
            return [float(array)]
        if array.ndim == 1:
            return [float(item) for item in array.tolist()]

        # 多维输出：取最后一列作为相关性分数
        return [float(item) for item in array[:, -1].tolist()]

    def _build_result(
        self,
        idx: int,
        score: float,
        vector_score: float,
        bm25_score: float,
        hybrid_score: float,
        rerank_score: float | None,
        retrieval_mode: str,
        multi_vector_scores: dict[str, float] | None = None,
        matched_vector_types: list[str] | None = None,
        best_vector_type: str | None = None,
    ) -> dict:
        """构建统一格式的检索结果字典。

        核心设计：返回父块内容（parent_content）作为主内容，
        同时保留子块内容（child_content）供调试或下游分析使用。
        元数据中包含完整的来源、层级和嵌入配置信息。
        """
        record = self.records[idx]
        child_content = record["content"]  # 子块内容（精确匹配用）
        parent_content = record.get("parent_content") or child_content  # 父块内容（展示用）
        return {
            # 核心分数
            "score": float(score),  # 当前模式的最终分数
            "vector_score": float(vector_score),  # 向量检索分数
            "bm25_score": float(bm25_score),  # BM25 检索分数
            "hybrid_score": float(hybrid_score),  # 混合融合分数
            "rerank_score": None if rerank_score is None else float(rerank_score),
            # 多向量详情
            "multi_vector_scores": multi_vector_scores or {},
            "matched_vector_types": matched_vector_types or [],
            "best_vector_type": best_vector_type,
            # 内容：父块为主，子块为辅助
            "content": parent_content,  # 返回父块完整内容（展示用）
            "child_content": child_content,  # 子块内容（调试用）
            # 来源信息
            "source": record["source"],
            "doc_id": record.get("doc_id"),
            "metadata": record["metadata"],
            "retrieval_mode": retrieval_mode,  # 检索模式标识
        }

    def _load_records(self) -> list[dict]:
        """从 metadata.json 加载所有 chunk 记录。

        包含兼容性处理：
        - 支持旧版格式（无父子分块）的自动迁移填充
        - 自动补全缺失的 metadata 字段（parent_id, child_chunk_id 等）
        - 规范化嵌入文本（缺失的用内容重新生成）

        Returns:
            规范化的 chunk 记录列表，每条记录格式与 _build_record 输出一致
        """
        # 文件不存在，返回空列表
        if not self.metadata_path.exists():
            return []
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        # 兼容两种格式：有 records 键的包装格式 / 直接的列表格式
        raw_records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(raw_records, list):
            return []

        records: list[dict] = []
        for fallback_index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, dict):
                continue
            # 跳过空内容的记录
            content = str(raw_record.get("content") or "")
            if not content.strip():
                continue
            source = str(raw_record.get("source") or "")
            # 获取 metadata 或初始化为空字典
            metadata = raw_record.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            # ── 兼容旧数据：推断缺失的字段值 ──
            # 父块索引（兼容旧字段名 chunk_index）
            parent_index = self._coerce_int(
                metadata.get("parent_index", metadata.get("chunk_index")),
                fallback=fallback_index,
            )
            # 子块在全局的序号
            child_chunk_index = self._coerce_int(
                metadata.get("child_chunk_index"),
                fallback=fallback_index,
            )
            # 子块在父块内的序号
            child_index = self._coerce_int(
                metadata.get("child_index"),
                fallback=0,
            )
            # 旧版 chunk_index 备用值
            legacy_chunk_index = self._coerce_int(
                metadata.get("chunk_index"),
                fallback=fallback_index,
            )

            # ── 解析或生成标识字段 ──
            # 文档 ID（支持多处来源的优先级链）
            doc_id = str(raw_record.get("doc_id") or metadata.get("doc_id") or self._document_id_for_source(source))
            # 源文件哈希
            source_hash = str(raw_record.get("source_hash") or metadata.get("source_hash") or "")
            # 子块内容哈希
            content_hash = str(raw_record.get("content_hash") or self._hash_text(content))
            # 父块内容
            parent_content = str(raw_record.get("parent_content") or content)
            # 父块内容哈希
            parent_content_hash = str(
                raw_record.get("parent_content_hash")
                or metadata.get("parent_content_hash")
                or self._hash_text(parent_content)
            )
            # 父块 ID（支持多处来源的优先级链）
            parent_id = str(
                raw_record.get("parent_id")
                or metadata.get("parent_id")
                or metadata.get("chunk_id")
                or f"{doc_id}:parent:{parent_index}:{parent_content_hash[:12]}"
            )
            # 子块 ID（支持多处来源的优先级链，兼容旧格式）
            child_chunk_id = str(
                raw_record.get("id")
                or metadata.get("child_chunk_id")
                or (
                    f"{doc_id}:parent:{parent_index}:child:{child_index}:{content_hash[:12]}"
                    if metadata.get("parent_id")
                    else metadata.get("chunk_id")
                )
                or f"{doc_id}:{legacy_chunk_index}:{content_hash[:12]}"
            )

            # ── 合并规范化 metadata ──
            merged_metadata = {
                **metadata,  # 保留原有字段
                "doc_id": doc_id,
                "chunk_id": parent_id,  # 兼容旧字段
                "chunk_index": parent_index,  # 兼容旧字段
                "parent_id": parent_id,
                "parent_index": parent_index,
                "parent_content_hash": parent_content_hash,
                "child_chunk_id": child_chunk_id,
                "child_chunk_index": child_chunk_index,
                "child_index": child_index,
            }
            if source_hash:
                merged_metadata["source_hash"] = source_hash
            if "chunking_strategy" in metadata:
                merged_metadata["chunking_strategy"] = metadata["chunking_strategy"]
            merged_metadata["embedding_strategy"] = str(metadata.get("embedding_strategy") or MULTI_VECTOR_STRATEGY)
            merged_metadata["embedding_types"] = list(MULTI_VECTOR_TYPES)
            merged_metadata["embedding_provider"] = str(metadata.get("embedding_provider") or "")
            merged_metadata["embedding_model"] = str(metadata.get("embedding_model") or "")
            merged_metadata["embedding_config_hash"] = str(metadata.get("embedding_config_hash") or "")

            # ── 规范化嵌入文本 ──
            embedding_texts = self._normalize_embedding_texts(
                raw_record.get("embedding_texts"),
                content,
            )

            # 构建规范化的记录
            records.append(
                {
                    "id": child_chunk_id,
                    "doc_id": doc_id,
                    "source": source,
                    "source_hash": source_hash,
                    "content_hash": content_hash,
                    "content": content,
                    "embedding_texts": embedding_texts,
                    "parent_id": parent_id,
                    "parent_content_hash": parent_content_hash,
                    "parent_content": parent_content,
                    "metadata": merged_metadata,
                }
            )
        return records

    def _load_documents_manifest(self) -> dict[str, dict]:
        """从 documents.json 加载文档清单。

        兼容两种格式：
        - 字典格式（推荐）：{"version": 1, "documents": {doc_id: info, ...}}
        - 列表格式（兼容旧版）：[{doc_id: ..., ...}, ...]

        Returns:
            {doc_id: document_info} 字典，document_info 已确保包含 doc_id 字段
        """
        if not self.documents_path.exists():
            return {}
        try:
            payload = json.loads(self.documents_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

        # 兼容包装格式和原始格式
        raw_documents = payload.get("documents") if isinstance(payload, dict) else payload
        documents: dict[str, dict] = {}
        if isinstance(raw_documents, dict):
            # 字典格式（推荐）
            for doc_id, info in raw_documents.items():
                if not isinstance(info, dict):
                    continue
                normalized = dict(info)
                normalized["doc_id"] = str(normalized.get("doc_id") or doc_id)
                documents[normalized["doc_id"]] = normalized
        elif isinstance(raw_documents, list):
            # 列表格式（兼容旧版）
            for info in raw_documents:
                if not isinstance(info, dict):
                    continue
                doc_id = str(info.get("doc_id") or "")
                if doc_id:
                    documents[doc_id] = dict(info)
        return documents

    def _reconcile_documents_manifest(self) -> bool:
        """协调文档清单：为 records 中存在但清单中缺失的文档补充条目。

        场景：手动修改 metadata.json 添加了 chunk，或将旧版本数据迁移过来时，
        documents.json 可能缺少对应的文档条目。此方法自动补全缺失的条目。

        Returns:
            是否有任何新条目被添加
        """
        # 按 doc_id 分组统计所有 chunk
        grouped_records: dict[str, list[dict]] = {}
        for record in self.records:
            grouped_records.setdefault(str(record["doc_id"]), []).append(record)

        changed = False
        for doc_id, records in grouped_records.items():
            # 已存在于清单中的跳过
            if doc_id in self.documents:
                continue

            first = records[0]
            # 推断分块策略（如果有记录则用记录的，否则标为旧版）
            record_strategy = first.get("metadata", {}).get("chunking_strategy")
            # 统计唯一的父块数量（去重 parent_id）
            parent_ids = {
                str(
                    record.get("parent_id")
                    or record.get("metadata", {}).get("parent_id")
                    or record.get("metadata", {}).get("chunk_id")
                )
                for record in records
            }

            # 创建补充的文档条目
            self.documents[doc_id] = {
                "doc_id": doc_id,
                "source": first["source"],
                "source_hash": first.get("source_hash") or "",
                "chunk_count": len(records),
                "parent_chunk_count": len(parent_ids),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "parent_chunk_size": (
                    self.parent_chunk_size if record_strategy == CHUNKING_STRATEGY else 0  # 旧版无父子分块
                ),
                "parent_chunk_overlap": (self.parent_chunk_overlap if record_strategy == CHUNKING_STRATEGY else 0),
                "chunking_strategy": record_strategy or "legacy_flat",
                "embedding_strategy": MULTI_VECTOR_STRATEGY,
                "embedding_types": list(MULTI_VECTOR_TYPES),
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model_name,
                "embedding_config_hash": self.embedding_config_hash,
                "indexed_at": utc_now(),
                "version": 1,
            }
            changed = True
        return changed

    def _persist_documents_manifest(self) -> None:
        """持久化文档清单到 documents.json。

        格式为：{"version": 1, "documents": {doc_id: {...}}}
        使用 indent=2 的 JSON 格式，方便人工查阅。
        """
        payload = {
            "version": DOCUMENTS_MANIFEST_VERSION,
            "documents": self.documents,
        }
        self.documents_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _document_id_for_path(self, path: Path) -> str:
        """根据文件路径生成全局唯一的文档 ID。"""
        return self._document_id_for_source(str(path))

    def _document_id_for_source(self, source: str) -> str:
        """根据源文件路径生成全局唯一的文档 ID。

        算法：将文件路径解析为绝对路径后取 SHA256 哈希的前 24 位。
        保证相同文件的 doc_id 在不同运行中一致，不同文件的 doc_id 碰撞概率极低。
        """
        path = Path(source).expanduser()
        if not path.is_absolute():
            # 相对路径以 RAG Server 项目根目录为基准解析
            path = self.base_dir.parent / path
        resolved = path.resolve(strict=False)
        return hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:24]

    def _hash_text(self, text: str) -> str:
        """计算文本的 SHA256 哈希值（全 64 位十六进制）。

        用于检测文档内容变更和 chunk 内容唯一标识。
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _coerce_int(self, value: Any, *, fallback: int) -> int:
        """安全地将任意值转换为整数，失败则返回 fallback。

        用于从旧版元数据中恢复整数字段，兼容缺失或非法的值。
        """
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _persist(self) -> None:
        """持久化所有索引和元数据到磁盘。

        三个文件的写入顺序：
        1. FAISS 向量索引 → faiss.index（二进制文件）
        2. 所有 chunk 记录 → metadata.json
        3. 文档清单 → documents.json

        最后使缓存版本失效，确保下次查询不会使用过期缓存。
        """
        # 持久化 FAISS 索引
        faiss.write_index(self.index, str(self.index_path))
        # 持久化 chunk 记录列表
        self.metadata_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 持久化文档清单
        self._persist_documents_manifest()
        # 使缓存版本失效（知识库已变更，缓存不再有效）
        self._invalidate_cache_version()
