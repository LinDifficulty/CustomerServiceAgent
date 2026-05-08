from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Plus
from sentence_transformers import CrossEncoder

from .trace_service import TraceRecorder, summarize_result

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import jieba

# 支持的文档格式集合，仅处理这些扩展名的文件。
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
DOCUMENTS_MANIFEST_VERSION = 1
CHUNKING_STRATEGY = "parent_child"
PARENT_CHILD_OVERFETCH_FACTOR = 5
MULTI_VECTOR_STRATEGY = "summary_keyword_semantic_v1"
MULTI_VECTOR_TYPES = ("summary", "keyword", "semantic")
DEFAULT_MULTI_VECTOR_WEIGHTS = {
    "summary": 0.25,
    "keyword": 0.25,
    "semantic": 0.5,
}
MAX_SUMMARY_EMBEDDING_CHARS = 240
MAX_KEYWORD_TERMS = 16
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RAGService:
    """轻量 RAG 服务。

    支持：
    1. DashScope 向量化 + FAISS 多向量检索
    2. BM25 关键词检索
    3. 向量 + BM25 混合召回
    4. Cross-Encoder 精排
    """

    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = "text-embedding-v4",
        embeddings: Any | None = None,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
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
    ) -> None:
        # 所有索引和元数据都保存在 data_dir，方便持久化复用。
        self.base_dir = Path(data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.base_dir / "faiss.index"
        self.metadata_path = self.base_dir / "metadata.json"
        self.documents_path = self.base_dir / "documents.json"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size or max(chunk_size * 3, chunk_size)
        self.parent_chunk_overlap = (
            min(chunk_overlap * 2, self.parent_chunk_size - 1)
            if parent_chunk_overlap is None
            else parent_chunk_overlap
        )
        self._validate_chunking_config(
            child_chunk_size=self.chunk_size,
            child_chunk_overlap=self.chunk_overlap,
            parent_chunk_size=self.parent_chunk_size,
            parent_chunk_overlap=self.parent_chunk_overlap,
        )

        # 向量模型默认使用 DashScope，也支持外部传入自定义 embeddings。
        self.embeddings = embeddings or DashScopeEmbeddings(model=model_name)

        # 重排序模型默认使用多语言 reranker，首次调用时再懒加载。
        self.reranker_model_name = reranker_model_name
        self.reranker = reranker
        self.reranker_device = reranker_device
        self.reranker_batch_size = reranker_batch_size
        self.default_use_bm25 = default_use_bm25
        self.default_use_rerank = default_use_rerank
        self.default_candidate_top_k = default_candidate_top_k
        self.trace_recorder = trace_recorder
        self.multi_vector_weights = self._normalize_multi_vector_weights(
            multi_vector_weights
        )

        self.documents = self._load_documents_manifest()
        self.records = self._load_records()
        documents_changed = self._reconcile_documents_manifest()
        self.bm25: BM25Plus | None = None
        self.vector_rows = self._build_vector_rows()
        self.index = self._load_or_create_index()
        self._rebuild_bm25()
        if documents_changed:
            self._persist_documents_manifest()

    def add_documents(
        self,
        file_paths: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        parent_chunk_size: int | None = None,
        parent_chunk_overlap: int | None = None,
    ) -> dict:
        """Upsert documents by source path.

        Re-indexes only documents whose content or chunking config changed.
        Keeping this method idempotent prevents repeated ingest runs from
        silently duplicating chunks.
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
        """Add new documents or replace changed documents in the local index."""
        actual_chunk_size = chunk_size or self.chunk_size
        actual_chunk_overlap = (
            self.chunk_overlap if chunk_overlap is None else chunk_overlap
        )
        actual_parent_chunk_size = parent_chunk_size or self.parent_chunk_size
        actual_parent_chunk_overlap = (
            self.parent_chunk_overlap
            if parent_chunk_overlap is None
            else parent_chunk_overlap
        )
        self._validate_chunking_config(
            child_chunk_size=actual_chunk_size,
            child_chunk_overlap=actual_chunk_overlap,
            parent_chunk_size=actual_parent_chunk_size,
            parent_chunk_overlap=actual_parent_chunk_overlap,
        )

        records_to_add: list[dict] = []
        added_sources: list[str] = []
        updated_sources: list[str] = []
        skipped_sources: list[str] = []
        deleted_chunks = 0
        added_parent_chunks = 0
        changed_documents = False

        for file_path in file_paths:
            path = Path(file_path)
            new_records, document_info = self._build_document_records(
                path,
                actual_chunk_size,
                actual_chunk_overlap,
                actual_parent_chunk_size,
                actual_parent_chunk_overlap,
            )
            doc_id = document_info["doc_id"]
            existing_info = self.documents.get(doc_id)

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

            existing_chunk_count = self._remove_document_records(doc_id)
            deleted_chunks += existing_chunk_count
            if existing_info is None:
                added_sources.append(document_info["source"])
                document_info["version"] = 1
            else:
                updated_sources.append(document_info["source"])
                document_info["version"] = int(existing_info.get("version") or 0) + 1

            self.documents[doc_id] = document_info
            records_to_add.extend(new_records)
            added_parent_chunks += int(document_info.get("parent_chunk_count") or 0)
            changed_documents = True

        if records_to_add:
            self.records.extend(records_to_add)

        if changed_documents:
            if deleted_chunks > 0:
                self._rebuild_vector_index()
            elif records_to_add:
                self._extend_vector_index(records_to_add)
            self._rebuild_bm25()
            self._persist()

        result = {
            "added_chunks": len(records_to_add),
            "added_parent_chunks": added_parent_chunks,
            "deleted_chunks": deleted_chunks,
            "sources": sorted({record["source"] for record in records_to_add}),
            "added_documents": sorted(added_sources),
            "updated_documents": sorted(updated_sources),
            "skipped_documents": sorted(skipped_sources),
        }
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
        """Replace one indexed document if the file content changed."""
        return self.upsert_documents(
            [file_path],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
        )

    def delete_document(self, document_ref: str) -> dict:
        """Delete one document by doc_id or source path and rebuild indexes."""
        doc_id = self._resolve_document_id(document_ref)
        if doc_id is None:
            result = {
                "deleted_chunks": 0,
                "document_id": None,
                "source": document_ref,
            }
            self._trace_event("rag.delete_document", result)
            return result

        document_info = self.documents.pop(doc_id, {})
        deleted_chunks = self._remove_document_records(doc_id)
        self._rebuild_vector_index()
        self._rebuild_bm25()
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
        """Upsert the given files and optionally remove documents not in the set."""
        result = self.upsert_documents(
            file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
        )
        if not remove_missing:
            return result

        desired_doc_ids = {
            self._document_id_for_path(Path(file_path))
            for file_path in file_paths
        }
        removed_sources: list[str] = []
        removed_chunks = 0
        for doc_id in sorted(set(self.documents) - desired_doc_ids):
            document_info = self.documents.pop(doc_id, {})
            removed_sources.append(str(document_info.get("source") or doc_id))
            removed_chunks += self._remove_document_records(doc_id)

        if removed_sources:
            self._rebuild_vector_index()
            self._rebuild_bm25()
            self._persist()

        result["removed_documents"] = removed_sources
        result["deleted_chunks"] += removed_chunks
        return result

    def list_documents(self) -> list[dict]:
        """Return indexed document metadata from the manifest."""
        return sorted(
            [dict(item) for item in self.documents.values()],
            key=lambda item: str(item.get("source", "")),
        )

    def _trace_event(self, name: str, payload: dict[str, Any]) -> None:
        if self.trace_recorder is None:
            return
        self.trace_recorder.event("rag", name, payload)

    def search_by_vector(self, query: str, top_k: int = 3) -> list[dict]:
        """仅使用向量召回。"""
        start = time.perf_counter()
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

        limit = self._candidate_limit(top_k)
        vector_details = self._vector_score_details_map(query, limit)

        candidate_results = []
        for idx, details in vector_details.items():
            vector_score = float(details["score"])
            candidate_results.append(
                self._build_result(
                    idx=idx,
                    score=vector_score,
                    vector_score=vector_score,
                    bm25_score=0.0,
                    hybrid_score=vector_score,
                    rerank_score=None,
                    retrieval_mode="multi_vector",
                    multi_vector_scores=details["scores"],
                    matched_vector_types=details["matched_vector_types"],
                    best_vector_type=details["best_vector_type"],
                )
            )

        candidate_results.sort(key=lambda item: item["score"], reverse=True)
        results = self._deduplicate_parent_results(candidate_results)[:top_k]
        self._trace_event(
            "rag.search_by_vector",
            {
                "query": query,
                "top_k": top_k,
                "candidate_count": len(candidate_results),
                "vector_row_count": len(self.vector_rows),
                "result_count": len(results),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "results": [
                    summarize_result(item, include_content=True) for item in results
                ],
            },
        )
        return results

    def search_by_bm25(
        self,
        query: str,
        top_k: int = 3,
        use_bm25: bool | None = None,
    ) -> list[dict]:
        """仅使用 BM25 关键词检索。"""
        start = time.perf_counter()
        actual_use_bm25 = (
            self.default_use_bm25 if use_bm25 is None else use_bm25
        )
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

        raw_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype="float32")
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

        normalized_scores = self._normalize_bm25_scores(raw_scores)
        top_indices = np.argsort(raw_scores)[::-1][: self._candidate_limit(top_k)]

        candidate_results = []
        for idx in top_indices:
            if raw_scores[idx] <= 0:
                continue
            bm25_score = float(normalized_scores[idx])
            candidate_results.append(
                self._build_result(
                    idx=int(idx),
                    score=bm25_score,
                    vector_score=0.0,
                    bm25_score=bm25_score,
                    hybrid_score=bm25_score,
                    rerank_score=None,
                    retrieval_mode="bm25",
                )
            )

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
                "results": [
                    summarize_result(item, include_content=True) for item in results
                ],
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
        """向量召回和 BM25 召回融合后的结果，不做精排。"""
        start = time.perf_counter()
        if not query.strip() or not self.records:
            self._trace_event(
                "rag.search_by_hybrid",
                {
                    "query": query,
                    "top_k": top_k,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
            return []

        actual_use_bm25 = (
            self.default_use_bm25 if use_bm25 is None else use_bm25
        )
        actual_vector_weight = 1.0 if not actual_use_bm25 else vector_weight
        actual_bm25_weight = bm25_weight if actual_use_bm25 else 0.0
        self._validate_weights(actual_vector_weight, actual_bm25_weight)

        limit = self._candidate_limit(top_k)
        vector_details = self._vector_score_details_map(query, limit)
        vector_scores = {
            idx: float(details["score"]) for idx, details in vector_details.items()
        }
        bm25_scores = self._bm25_score_map(query, limit) if actual_use_bm25 else {}
        candidate_ids = set(vector_scores) | set(bm25_scores)

        ranked_results = []
        for idx in candidate_ids:
            vector_score = vector_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
            details = vector_details.get(
                idx,
                {
                    "scores": {},
                    "matched_vector_types": [],
                    "best_vector_type": None,
                },
            )
            hybrid_score = (
                actual_vector_weight * vector_score
                + actual_bm25_weight * bm25_score
            )
            ranked_results.append(
                self._build_result(
                    idx=idx,
                    score=hybrid_score,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
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
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "results": [
                    summarize_result(item, include_content=True) for item in results
                ],
            },
        )
        return results

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """使用 Cross-Encoder 对候选结果做精排。"""
        start = time.perf_counter()
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

        reranker = self._get_reranker()
        pairs = [(query, item["content"]) for item in candidates]
        raw_scores = reranker.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
        )
        rerank_scores = self._coerce_rerank_scores(raw_scores)

        reranked = []
        for item, rerank_score in zip(candidates, rerank_scores, strict=False):
            merged = dict(item)
            merged["score"] = float(rerank_score)
            merged["rerank_score"] = float(rerank_score)
            merged["retrieval_mode"] = "hybrid_rerank"
            reranked.append(merged)

        reranked.sort(key=lambda item: item["score"], reverse=True)
        results = reranked if top_k is None else reranked[:top_k]
        self._trace_event(
            "rag.rerank",
            {
                "query": query,
                "reranker_model_name": self.reranker_model_name,
                "candidate_count": len(candidates),
                "top_k": top_k,
                "result_count": len(results),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "candidates": [
                    summarize_result(item, include_content=True)
                    for item in candidates
                ],
                "results": [
                    summarize_result(item, include_content=True) for item in results
                ],
            },
        )
        return results

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
        """默认搜索入口。

        先做混合召回，再按配置决定是否执行 Cross-Encoder 精排。
        """
        start = time.perf_counter()
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

        actual_use_rerank = (
            self.default_use_rerank if use_rerank is None else use_rerank
        )
        actual_candidate_top_k = candidate_top_k or self.default_candidate_top_k
        actual_candidate_top_k = max(actual_candidate_top_k, top_k)

        hybrid_candidates = self.search_by_hybrid(
            query=query,
            top_k=actual_candidate_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            use_bm25=use_bm25,
        )

        if not actual_use_rerank:
            results = hybrid_candidates[:top_k]
        else:
            results = self.rerank(
                query=query,
                candidates=hybrid_candidates,
                top_k=top_k,
            )

        self._trace_event(
            "rag.search",
            {
                "query": query,
                "top_k": top_k,
                "candidate_top_k": actual_candidate_top_k,
                "use_bm25": self.default_use_bm25 if use_bm25 is None else use_bm25,
                "use_rerank": actual_use_rerank,
                "candidate_count": len(hybrid_candidates),
                "result_count": len(results),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "results": [
                    summarize_result(item, include_content=True) for item in results
                ],
            },
        )
        return results

    def reset(self) -> None:
        """清空 FAISS、BM25 和元数据。"""
        self.records = []
        self.documents = {}
        self.vector_rows = []
        self.index = self._create_index()
        self._rebuild_bm25()
        self._persist()

    def _build_document_records(
        self,
        path: Path,
        chunk_size: int,
        chunk_overlap: int,
        parent_chunk_size: int,
        parent_chunk_overlap: int,
    ) -> tuple[list[dict], dict]:
        text = self._read_document(path)
        source = str(path)
        doc_id = self._document_id_for_path(path)
        source_hash = self._hash_text(text)
        parent_chunks = self._split_text(
            text,
            parent_chunk_size,
            parent_chunk_overlap,
        )
        indexed_at = _utc_now()

        records: list[dict] = []
        for parent_index, parent_content in enumerate(parent_chunks):
            parent_content_hash = self._hash_text(parent_content)
            parent_id = f"{doc_id}:parent:{parent_index}:{parent_content_hash[:12]}"
            child_chunks = self._split_text(parent_content, chunk_size, chunk_overlap)
            for child_index, child_content in enumerate(child_chunks):
                records.append(
                    self._build_record(
                        doc_id=doc_id,
                        source=source,
                        source_hash=source_hash,
                        child_content=child_content,
                        child_chunk_index=len(records),
                        child_index=child_index,
                        parent_id=parent_id,
                        parent_index=parent_index,
                        parent_content=parent_content,
                        parent_content_hash=parent_content_hash,
                    )
                )

        document_info = {
            "doc_id": doc_id,
            "source": source,
            "source_hash": source_hash,
            "chunk_count": len(records),
            "parent_chunk_count": len(parent_chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "parent_chunk_size": parent_chunk_size,
            "parent_chunk_overlap": parent_chunk_overlap,
            "chunking_strategy": CHUNKING_STRATEGY,
            "embedding_strategy": MULTI_VECTOR_STRATEGY,
            "embedding_types": list(MULTI_VECTOR_TYPES),
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
        content_hash = self._hash_text(child_content)
        child_chunk_id = (
            f"{doc_id}:parent:{parent_index}:child:{child_index}:{content_hash[:12]}"
        )
        return {
            "id": child_chunk_id,
            "doc_id": doc_id,
            "source": source,
            "source_hash": source_hash,
            "content_hash": content_hash,
            "content": child_content,
            "embedding_texts": self._build_embedding_texts(child_content),
            "parent_id": parent_id,
            "parent_content_hash": parent_content_hash,
            "parent_content": parent_content,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": parent_id,
                "chunk_index": parent_index,
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
        if existing_info is None:
            return False
        return (
            existing_info.get("source_hash") == new_info.get("source_hash")
            and int(existing_info.get("chunk_size") or 0) == chunk_size
            and int(existing_info.get("chunk_overlap") or 0) == chunk_overlap
            and int(existing_info.get("parent_chunk_size") or 0) == parent_chunk_size
            and int(existing_info.get("parent_chunk_overlap") or 0)
            == parent_chunk_overlap
            and existing_info.get("chunking_strategy") == CHUNKING_STRATEGY
            and existing_info.get("embedding_strategy") == MULTI_VECTOR_STRATEGY
        )

    def _remove_document_records(self, doc_id: str) -> int:
        before = len(self.records)
        self.records = [
            record
            for record in self.records
            if str(record.get("doc_id") or record.get("metadata", {}).get("doc_id"))
            != doc_id
        ]
        return before - len(self.records)

    def _resolve_document_id(self, document_ref: str) -> str | None:
        if document_ref in self.documents:
            return document_ref

        path_doc_id = self._document_id_for_path(Path(document_ref))
        if path_doc_id in self.documents:
            return path_doc_id

        ref_path = Path(document_ref).expanduser()
        if not ref_path.is_absolute():
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
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if path.suffix.lower() in {".txt", ".md"}:
            return path.read_text(encoding="utf-8").strip()

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        if not text:
            return []

        # 优先按段落和中文标点切，能比纯字符切片保留更多自然语义边界。
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        return splitter.split_text(text)

    def _build_embedding_texts(self, content: str) -> dict[str, str]:
        normalized = self._normalize_whitespace(content)
        summary = self._chunk_summary_text(normalized)
        keywords = self._chunk_keyword_text(normalized)
        return {
            "summary": f"摘要: {summary or normalized}",
            "keyword": f"关键词: {keywords or summary or normalized}",
            "semantic": f"语义: {normalized}",
        }

    def _normalize_embedding_texts(self, value: Any, content: str) -> dict[str, str]:
        defaults = self._build_embedding_texts(content)
        if not isinstance(value, dict):
            return defaults

        normalized = {}
        for vector_type in MULTI_VECTOR_TYPES:
            text = str(value.get(vector_type) or "").strip()
            normalized[vector_type] = text or defaults[vector_type]
        return normalized

    def _build_vector_rows(self) -> list[dict]:
        rows: list[dict] = []
        for record_index, record in enumerate(self.records):
            embedding_texts = self._normalize_embedding_texts(
                record.get("embedding_texts"),
                str(record.get("content") or ""),
            )
            record["embedding_texts"] = embedding_texts
            for vector_type in MULTI_VECTOR_TYPES:
                rows.append(
                    {
                        "record_index": record_index,
                        "vector_type": vector_type,
                        "text": embedding_texts[vector_type],
                    }
                )
        return rows

    def _chunk_summary_text(self, text: str) -> str:
        if not text:
            return ""

        sentences = [
            item.strip()
            for item in re.split(r"(?<=[。！？；.!?;])\s*|\n+", text)
            if item.strip()
        ]
        summary = " ".join(sentences[:2]) if sentences else text
        if len(summary) <= MAX_SUMMARY_EMBEDDING_CHARS:
            return summary
        return summary[:MAX_SUMMARY_EMBEDDING_CHARS].rstrip()

    def _chunk_keyword_text(self, text: str) -> str:
        keywords = self._extract_keywords(text)
        return " ".join(keywords)

    def _extract_keywords(self, text: str) -> list[str]:
        first_seen: dict[str, int] = {}
        tokens: list[str] = []
        for token in self._tokenize(text):
            if not self._is_keyword_token(token):
                continue
            if token not in first_seen:
                first_seen[token] = len(first_seen)
            tokens.append(token)

        if not tokens:
            return []

        counts = Counter(tokens)
        ranked = sorted(
            counts,
            key=lambda token: (-counts[token], first_seen[token], token),
        )
        return ranked[:MAX_KEYWORD_TERMS]

    def _is_keyword_token(self, token: str) -> bool:
        if token in KEYWORD_STOPWORDS:
            return False
        if re.fullmatch(r"[\W_]+", token):
            return False
        return any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in token)

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.embeddings.embed_documents(texts)
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        return matrix

    def _embed_query(self, text: str) -> np.ndarray:
        if hasattr(self.embeddings, "embed_query"):
            vector = self.embeddings.embed_query(text)
        else:
            vector = self.embeddings.embed_documents([text])[0]
        matrix = np.asarray([vector], dtype="float32")
        faiss.normalize_L2(matrix)
        return matrix

    def _load_or_create_index(self) -> faiss.Index:
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.ntotal == len(self.vector_rows):
                return index
            return self._rebuild_and_persist_vector_index()
        if self.vector_rows:
            return self._rebuild_and_persist_vector_index()
        return self._create_index()

    def _create_index(self) -> faiss.Index:
        return faiss.IndexFlatIP(0)

    def _rebuild_and_persist_vector_index(self) -> faiss.Index:
        index = self._rebuild_vector_index()
        faiss.write_index(index, str(self.index_path))
        return index

    def _rebuild_vector_index(self) -> faiss.Index:
        self.vector_rows = self._build_vector_rows()
        if not self.vector_rows:
            self.index = self._create_index()
            return self.index

        vectors = self._embed_texts([row["text"] for row in self.vector_rows])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self.index

    def _extend_vector_index(self, new_records: list[dict]) -> None:
        """Embed only *new_records* and append to the existing FAISS index."""
        new_rows: list[dict] = []
        base_record_index = len(self.records) - len(new_records)
        for offset, record in enumerate(new_records):
            record_index = base_record_index + offset
            embedding_texts = self._normalize_embedding_texts(
                record.get("embedding_texts"),
                str(record.get("content") or ""),
            )
            record["embedding_texts"] = embedding_texts
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

        vectors = self._embed_texts([row["text"] for row in new_rows])
        if self.index.d == 0:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.vector_rows.extend(new_rows)

    def _rebuild_bm25(self) -> None:
        # BM25 不单独持久化，启动时根据已有 chunk 重建即可。
        tokenized_corpus = [
            self._tokenize(record["content"])
            for record in self.records
            if record["content"].strip()
        ]
        # 小语料场景下，BM25Plus 比 BM25Okapi 更容易得到稳定的关键词分数。
        self.bm25 = BM25Plus(tokenized_corpus) if tokenized_corpus else None

    def _tokenize(self, text: str) -> list[str]:
        # jieba 对中文关键词检索比简单 split 更合适。
        return [
            token.lower().strip()
            for token in jieba.lcut_for_search(text)
            if token.strip()
        ]

    def _vector_score_map(self, query: str, limit: int) -> dict[int, float]:
        return {
            idx: float(details["score"])
            for idx, details in self._vector_score_details_map(query, limit).items()
        }

    def _vector_score_details_map(self, query: str, limit: int) -> dict[int, dict]:
        if self.index.ntotal == 0:
            return {}

        row_limit = self._vector_row_limit(limit)
        if row_limit <= 0:
            return {}

        scores, indices = self.index.search(self._embed_query(query), row_limit)
        hits: dict[int, dict] = {}
        for raw_score, row_idx in zip(scores[0], indices[0], strict=False):
            if row_idx < 0 or row_idx >= len(self.vector_rows):
                continue

            vector_row = self.vector_rows[int(row_idx)]
            record_idx = int(vector_row["record_index"])
            vector_type = str(vector_row["vector_type"])
            vector_score = self._normalize_vector_score(float(raw_score))
            hit = hits.setdefault(
                record_idx,
                {
                    "scores": {},
                    "row_indices": {},
                },
            )
            previous = hit["scores"].get(vector_type)
            if previous is None or vector_score > previous:
                hit["scores"][vector_type] = vector_score
                hit["row_indices"][vector_type] = int(row_idx)

        result: dict[int, dict] = {}
        for record_idx, hit in hits.items():
            vector_scores = hit["scores"]
            matched_vector_types = sorted(
                vector_scores,
                key=lambda item: vector_scores[item],
                reverse=True,
            )
            result[record_idx] = {
                "score": self._aggregate_multi_vector_scores(vector_scores),
                "scores": {
                    vector_type: float(vector_scores[vector_type])
                    for vector_type in MULTI_VECTOR_TYPES
                    if vector_type in vector_scores
                },
                "matched_vector_types": matched_vector_types,
                "best_vector_type": (
                    matched_vector_types[0] if matched_vector_types else None
                ),
                "row_indices": hit["row_indices"],
            }
        return result

    def _aggregate_multi_vector_scores(self, scores: dict[str, float]) -> float:
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

        if matched_weight <= 0:
            return max(float(score) for score in scores.values())
        return max(0.0, min(1.0, weighted_total / matched_weight))

    def _bm25_score_map(self, query: str, limit: int) -> dict[int, float]:
        if self.bm25 is None:
            return {}

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        raw_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype="float32")
        if raw_scores.size == 0 or float(raw_scores.max()) <= 0:
            return {}

        normalized_scores = self._normalize_bm25_scores(raw_scores)
        top_indices = np.argsort(raw_scores)[::-1][:limit]
        return {
            int(idx): float(normalized_scores[idx])
            for idx in top_indices
            if raw_scores[idx] > 0
        }

    def _normalize_vector_score(self, score: float) -> float:
        # 内积分数近似落在 [-1, 1]，缩放到 [0, 1] 后更方便与 BM25 融合。
        return max(0.0, min(1.0, (score + 1) / 2))

    def _normalize_bm25_scores(self, scores: np.ndarray) -> np.ndarray:
        max_score = float(scores.max())
        if max_score <= 0:
            return np.zeros_like(scores)
        return scores / max_score

    def _candidate_limit(self, top_k: int) -> int:
        requested = max(1, top_k)
        return min(requested * PARENT_CHILD_OVERFETCH_FACTOR, len(self.records))

    def _vector_row_limit(self, record_limit: int) -> int:
        if not self.vector_rows:
            return 0
        requested = max(1, record_limit) * len(MULTI_VECTOR_TYPES)
        return min(requested, len(self.vector_rows))

    def _deduplicate_parent_results(self, results: list[dict]) -> list[dict]:
        deduplicated: list[dict] = []
        seen_parent_keys: set[tuple[str, str]] = set()
        for item in results:
            parent_key = self._parent_result_key(item)
            if parent_key in seen_parent_keys:
                continue
            seen_parent_keys.add(parent_key)
            deduplicated.append(item)
        return deduplicated

    def _parent_result_key(self, item: dict) -> tuple[str, str]:
        metadata = item.get("metadata") or {}
        doc_id = str(item.get("doc_id") or metadata.get("doc_id") or "")
        parent_id = str(
            metadata.get("parent_id")
            or metadata.get("chunk_id")
            or metadata.get("chunk_index")
            or ""
        )
        return doc_id, parent_id

    def _validate_weights(self, vector_weight: float, bm25_weight: float) -> None:
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("vector_weight and bm25_weight must be non-negative")
        if vector_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one retrieval weight must be greater than 0")

    def _normalize_multi_vector_weights(
        self,
        weights: dict[str, float] | None,
    ) -> dict[str, float]:
        normalized = dict(DEFAULT_MULTI_VECTOR_WEIGHTS)
        if weights is None:
            return normalized

        unknown = sorted(set(weights) - set(MULTI_VECTOR_TYPES))
        if unknown:
            raise ValueError(
                "Unknown multi-vector weight(s): " + ", ".join(unknown)
            )

        for vector_type, raw_weight in weights.items():
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("multi-vector weights must be non-negative")
            normalized[vector_type] = weight

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
        if child_chunk_size <= 0 or parent_chunk_size <= 0:
            raise ValueError("chunk_size and parent_chunk_size must be positive")
        if child_chunk_overlap < 0 or parent_chunk_overlap < 0:
            raise ValueError("chunk overlaps must be non-negative")
        if child_chunk_overlap >= child_chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if parent_chunk_overlap >= parent_chunk_size:
            raise ValueError(
                "parent_chunk_overlap must be smaller than parent_chunk_size"
            )
        if child_chunk_size > parent_chunk_size:
            raise ValueError(
                "chunk_size must be smaller than or equal to parent_chunk_size"
            )

    def _get_reranker(self) -> Any:
        # 懒加载 reranker，避免只做入库时也下载和初始化 Cross-Encoder。
        if self.reranker is None:
            self.reranker = CrossEncoder(
                self.reranker_model_name,
                device=self.reranker_device,
                trust_remote_code=True,
            )
        return self.reranker

    def _coerce_rerank_scores(self, scores: Any) -> list[float]:
        array = np.asarray(scores, dtype="float32")
        if array.ndim == 0:
            return [float(array)]
        if array.ndim == 1:
            return [float(item) for item in array.tolist()]

        # 某些模型可能返回多维输出，这里默认取最后一列作为相关性分数。
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
        record = self.records[idx]
        child_content = record["content"]
        parent_content = record.get("parent_content") or child_content
        return {
            "score": float(score),
            "vector_score": float(vector_score),
            "bm25_score": float(bm25_score),
            "hybrid_score": float(hybrid_score),
            "rerank_score": None if rerank_score is None else float(rerank_score),
            "multi_vector_scores": multi_vector_scores or {},
            "matched_vector_types": matched_vector_types or [],
            "best_vector_type": best_vector_type,
            "content": parent_content,
            "child_content": child_content,
            "source": record["source"],
            "doc_id": record.get("doc_id"),
            "metadata": record["metadata"],
            "retrieval_mode": retrieval_mode,
        }

    def _load_records(self) -> list[dict]:
        if not self.metadata_path.exists():
            return []
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        raw_records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(raw_records, list):
            return []

        records: list[dict] = []
        for fallback_index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, dict):
                continue
            content = str(raw_record.get("content") or "")
            if not content.strip():
                continue
            source = str(raw_record.get("source") or "")
            metadata = raw_record.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            parent_index = self._coerce_int(
                metadata.get("parent_index", metadata.get("chunk_index")),
                fallback=fallback_index,
            )
            child_chunk_index = self._coerce_int(
                metadata.get("child_chunk_index"),
                fallback=fallback_index,
            )
            child_index = self._coerce_int(
                metadata.get("child_index"),
                fallback=0,
            )
            legacy_chunk_index = self._coerce_int(
                metadata.get("chunk_index"),
                fallback=fallback_index,
            )
            doc_id = str(
                raw_record.get("doc_id")
                or metadata.get("doc_id")
                or self._document_id_for_source(source)
            )
            source_hash = str(
                raw_record.get("source_hash")
                or metadata.get("source_hash")
                or ""
            )
            content_hash = str(
                raw_record.get("content_hash") or self._hash_text(content)
            )
            parent_content = str(raw_record.get("parent_content") or content)
            parent_content_hash = str(
                raw_record.get("parent_content_hash")
                or metadata.get("parent_content_hash")
                or self._hash_text(parent_content)
            )
            parent_id = str(
                raw_record.get("parent_id")
                or metadata.get("parent_id")
                or metadata.get("chunk_id")
                or f"{doc_id}:parent:{parent_index}:{parent_content_hash[:12]}"
            )
            child_chunk_id = str(
                raw_record.get("id")
                or metadata.get("child_chunk_id")
                or (
                    f"{doc_id}:parent:{parent_index}:child:"
                    f"{child_index}:{content_hash[:12]}"
                    if metadata.get("parent_id")
                    else metadata.get("chunk_id")
                )
                or f"{doc_id}:{legacy_chunk_index}:{content_hash[:12]}"
            )
            merged_metadata = {
                **metadata,
                "doc_id": doc_id,
                "chunk_id": parent_id,
                "chunk_index": parent_index,
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
            merged_metadata["embedding_strategy"] = str(
                metadata.get("embedding_strategy") or MULTI_VECTOR_STRATEGY
            )
            merged_metadata["embedding_types"] = list(MULTI_VECTOR_TYPES)

            embedding_texts = self._normalize_embedding_texts(
                raw_record.get("embedding_texts"),
                content,
            )

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
        if not self.documents_path.exists():
            return {}
        try:
            payload = json.loads(self.documents_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

        raw_documents = (
            payload.get("documents") if isinstance(payload, dict) else payload
        )
        documents: dict[str, dict] = {}
        if isinstance(raw_documents, dict):
            for doc_id, info in raw_documents.items():
                if not isinstance(info, dict):
                    continue
                normalized = dict(info)
                normalized["doc_id"] = str(normalized.get("doc_id") or doc_id)
                documents[normalized["doc_id"]] = normalized
        elif isinstance(raw_documents, list):
            for info in raw_documents:
                if not isinstance(info, dict):
                    continue
                doc_id = str(info.get("doc_id") or "")
                if doc_id:
                    documents[doc_id] = dict(info)
        return documents

    def _reconcile_documents_manifest(self) -> bool:
        grouped_records: dict[str, list[dict]] = {}
        for record in self.records:
            grouped_records.setdefault(str(record["doc_id"]), []).append(record)

        changed = False
        for doc_id, records in grouped_records.items():
            if doc_id in self.documents:
                continue
            first = records[0]
            record_strategy = first.get("metadata", {}).get("chunking_strategy")
            parent_ids = {
                str(
                    record.get("parent_id")
                    or record.get("metadata", {}).get("parent_id")
                    or record.get("metadata", {}).get("chunk_id")
                )
                for record in records
            }
            self.documents[doc_id] = {
                "doc_id": doc_id,
                "source": first["source"],
                "source_hash": first.get("source_hash") or "",
                "chunk_count": len(records),
                "parent_chunk_count": len(parent_ids),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "parent_chunk_size": (
                    self.parent_chunk_size
                    if record_strategy == CHUNKING_STRATEGY
                    else 0
                ),
                "parent_chunk_overlap": (
                    self.parent_chunk_overlap
                    if record_strategy == CHUNKING_STRATEGY
                    else 0
                ),
                "chunking_strategy": record_strategy or "legacy_flat",
                "embedding_strategy": MULTI_VECTOR_STRATEGY,
                "embedding_types": list(MULTI_VECTOR_TYPES),
                "indexed_at": _utc_now(),
                "version": 1,
            }
            changed = True
        return changed

    def _persist_documents_manifest(self) -> None:
        payload = {
            "version": DOCUMENTS_MANIFEST_VERSION,
            "documents": self.documents,
        }
        self.documents_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _document_id_for_path(self, path: Path) -> str:
        return self._document_id_for_source(str(path))

    def _document_id_for_source(self, source: str) -> str:
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = self.base_dir.parent / path
        resolved = path.resolve(strict=False)
        return hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:24]

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _coerce_int(self, value: Any, *, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._persist_documents_manifest()
