from __future__ import annotations

import hashlib
import json
import time
import warnings
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RAGService:
    """轻量 RAG 服务。

    支持：
    1. DashScope 向量化 + FAISS 向量检索
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
        default_use_rerank: bool = True,
        default_candidate_top_k: int = 20,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        # 所有索引和元数据都保存在 data_dir，方便持久化复用。
        self.base_dir = Path(data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.base_dir / "faiss.index"
        self.metadata_path = self.base_dir / "metadata.json"
        self.documents_path = self.base_dir / "documents.json"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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

        self.documents = self._load_documents_manifest()
        self.records = self._load_records()
        self._reconcile_documents_manifest()
        self.bm25: BM25Plus | None = None
        self.index = self._load_or_create_index()
        self._rebuild_bm25()

    def add_documents(
        self,
        file_paths: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
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
        )

    def upsert_documents(
        self,
        file_paths: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict:
        """Add new documents or replace changed documents in the local index."""
        actual_chunk_size = chunk_size or self.chunk_size
        actual_chunk_overlap = (
            self.chunk_overlap if chunk_overlap is None else chunk_overlap
        )
        if actual_chunk_overlap >= actual_chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        records_to_add: list[dict] = []
        added_sources: list[str] = []
        updated_sources: list[str] = []
        skipped_sources: list[str] = []
        deleted_chunks = 0
        changed_documents = False

        for file_path in file_paths:
            path = Path(file_path)
            new_records, document_info = self._build_document_records(
                path,
                actual_chunk_size,
                actual_chunk_overlap,
            )
            doc_id = document_info["doc_id"]
            existing_info = self.documents.get(doc_id)

            if self._document_is_unchanged(
                existing_info,
                document_info,
                actual_chunk_size,
                actual_chunk_overlap,
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
            changed_documents = True

        if records_to_add:
            self.records.extend(records_to_add)

        if changed_documents:
            self._rebuild_vector_index()
            self._rebuild_bm25()
            self._persist()

        result = {
            "added_chunks": len(records_to_add),
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
    ) -> dict:
        """Replace one indexed document if the file content changed."""
        return self.upsert_documents(
            [file_path],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
    ) -> dict:
        """Upsert the given files and optionally remove documents not in the set."""
        result = self.upsert_documents(
            file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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

        limit = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(self._embed_texts([query]), limit)

        results = []
        for raw_score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            vector_score = self._normalize_vector_score(float(raw_score))
            results.append(
                self._build_result(
                    idx=int(idx),
                    score=vector_score,
                    vector_score=vector_score,
                    bm25_score=0.0,
                    hybrid_score=vector_score,
                    rerank_score=None,
                    retrieval_mode="vector",
                )
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        self._trace_event(
            "rag.search_by_vector",
            {
                "query": query,
                "top_k": top_k,
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
        top_indices = np.argsort(raw_scores)[::-1][: min(top_k, len(raw_scores))]

        results = []
        for idx in top_indices:
            if raw_scores[idx] <= 0:
                continue
            bm25_score = float(normalized_scores[idx])
            results.append(
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

        results.sort(key=lambda item: item["score"], reverse=True)
        self._trace_event(
            "rag.search_by_bm25",
            {
                "query": query,
                "top_k": top_k,
                "use_bm25": actual_use_bm25,
                "tokens": query_tokens,
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

        limit = min(top_k, len(self.records))
        vector_scores = self._vector_score_map(query, limit)
        bm25_scores = self._bm25_score_map(query, limit) if actual_use_bm25 else {}
        candidate_ids = set(vector_scores) | set(bm25_scores)

        ranked_results = []
        for idx in candidate_ids:
            vector_score = vector_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
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
                    retrieval_mode="hybrid" if actual_use_bm25 else "vector",
                )
            )

        ranked_results.sort(key=lambda item: item["score"], reverse=True)
        results = ranked_results[:top_k]
        self._trace_event(
            "rag.search_by_hybrid",
            {
                "query": query,
                "top_k": top_k,
                "vector_weight": actual_vector_weight,
                "bm25_weight": actual_bm25_weight,
                "use_bm25": actual_use_bm25,
                "candidate_count": len(candidate_ids),
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
        self.index = self._create_index()
        self._rebuild_bm25()
        self._persist()

    def _build_document_records(
        self,
        path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> tuple[list[dict], dict]:
        text = self._read_document(path)
        source = str(path)
        doc_id = self._document_id_for_path(path)
        source_hash = self._hash_text(text)
        chunks = self._split_text(text, chunk_size, chunk_overlap)
        indexed_at = _utc_now()

        records = [
            self._build_record(
                doc_id=doc_id,
                source=source,
                source_hash=source_hash,
                content=content,
                chunk_index=chunk_index,
            )
            for chunk_index, content in enumerate(chunks)
        ]
        document_info = {
            "doc_id": doc_id,
            "source": source,
            "source_hash": source_hash,
            "chunk_count": len(records),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
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
        content: str,
        chunk_index: int,
    ) -> dict:
        content_hash = self._hash_text(content)
        chunk_id = f"{doc_id}:{chunk_index}:{content_hash[:12]}"
        return {
            "id": chunk_id,
            "doc_id": doc_id,
            "source": source,
            "source_hash": source_hash,
            "content_hash": content_hash,
            "content": content,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "source_hash": source_hash,
            },
        }

    def _document_is_unchanged(
        self,
        existing_info: dict | None,
        new_info: dict,
        chunk_size: int,
        chunk_overlap: int,
    ) -> bool:
        if existing_info is None:
            return False
        return (
            existing_info.get("source_hash") == new_info.get("source_hash")
            and int(existing_info.get("chunk_size") or 0) == chunk_size
            and int(existing_info.get("chunk_overlap") or 0) == chunk_overlap
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

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.embeddings.embed_documents(texts)
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        return matrix

    def _load_or_create_index(self) -> faiss.Index:
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.ntotal == len(self.records):
                return index
            return self._rebuild_vector_index()
        if self.records:
            return self._rebuild_vector_index()
        return self._create_index()

    def _create_index(self) -> faiss.Index:
        dimension = len(self.embeddings.embed_query("test"))
        return faiss.IndexFlatIP(dimension)

    def _rebuild_vector_index(self) -> faiss.Index:
        if not self.records:
            self.index = self._create_index()
            return self.index

        vectors = self._embed_texts([record["content"] for record in self.records])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self.index

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
        if self.index.ntotal == 0:
            return {}

        scores, indices = self.index.search(self._embed_texts([query]), limit)
        result: dict[int, float] = {}
        for raw_score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            result[int(idx)] = self._normalize_vector_score(float(raw_score))
        return result

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

    def _validate_weights(self, vector_weight: float, bm25_weight: float) -> None:
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("vector_weight and bm25_weight must be non-negative")
        if vector_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one retrieval weight must be greater than 0")

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
    ) -> dict:
        record = self.records[idx]
        return {
            "score": float(score),
            "vector_score": float(vector_score),
            "bm25_score": float(bm25_score),
            "hybrid_score": float(hybrid_score),
            "rerank_score": None if rerank_score is None else float(rerank_score),
            "content": record["content"],
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

            chunk_index = self._coerce_int(
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
            chunk_id = str(
                raw_record.get("id")
                or metadata.get("chunk_id")
                or f"{doc_id}:{chunk_index}:{content_hash[:12]}"
            )
            merged_metadata = {
                **metadata,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
            }
            if source_hash:
                merged_metadata["source_hash"] = source_hash

            records.append(
                {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "source": source,
                    "source_hash": source_hash,
                    "content_hash": content_hash,
                    "content": content,
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

    def _reconcile_documents_manifest(self) -> None:
        grouped_records: dict[str, list[dict]] = {}
        for record in self.records:
            grouped_records.setdefault(str(record["doc_id"]), []).append(record)

        for doc_id, records in grouped_records.items():
            if doc_id in self.documents:
                continue
            first = records[0]
            self.documents[doc_id] = {
                "doc_id": doc_id,
                "source": first["source"],
                "source_hash": first.get("source_hash") or "",
                "chunk_count": len(records),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "indexed_at": _utc_now(),
                "version": 1,
            }

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
