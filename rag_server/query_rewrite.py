from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .cache_service import JsonCache
from .llm_retry import LLMRetryPolicy, ainvoke_with_retry, invoke_with_retry
from .model_factory import DEFAULT_CHAT_MODEL, DEFAULT_CHAT_PROVIDER, create_chat_model
from .rag_service import RAGService
from .trace_service import TraceRecorder, summarize_result
from .utils import coerce_message_content, parse_json_object


@dataclass
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    search_queries: list[str]
    notes: list[str]
    raw_response: str


class LLMQueryRewriter:
    """Use the same LLM family as the agent to rewrite retrieval queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_CHAT_MODEL,
        provider: str = DEFAULT_CHAT_PROVIDER,
        model_kwargs: dict[str, Any] | None = None,
        model: Any | None = None,
        trace_recorder: TraceRecorder | None = None,
        retry_policy: LLMRetryPolicy | None = None,
        cache: JsonCache | None = None,
        cache_ttl_s: int = 86400,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs or {})
        self.model = model or create_chat_model(
            provider=provider,
            model_name=model_name,
            **self.model_kwargs,
        )
        self.trace_recorder = trace_recorder
        self.retry_policy = retry_policy or LLMRetryPolicy()
        self.cache = cache
        self.cache_ttl_s = cache_ttl_s
        self.system_prompt = SystemMessage(
            content=(
                "你是一个RAG检索改写器，只负责把用户问题改写成更适合知识库检索的查询。"
                "不要回答问题，不要补造事实。"
                "必须保留用户给出的硬约束，包括数字、单位、颜色、材质、季节、场景、体型和诉求。"
                "可以补充常见同义词、标准表达和更完整的检索短语。"
                "输出必须是JSON对象，格式如下："
                '{"rewritten_query":"...","search_queries":["..."],"notes":["..."]}'
                "其中 search_queries 返回1到3条中文检索语句，按推荐顺序排列。"
                "如果原问题已经很适合检索，rewritten_query 可以接近原句，但要更规范。"
            )
        )

    def rewrite(self, query: str) -> QueryRewriteResult:
        start = time.perf_counter()
        cached = self._read_cached_result(query, start)
        if cached is not None:
            return cached

        messages = self._build_messages(query)
        response = invoke_with_retry(
            lambda: self.model.invoke(messages),
            retry_policy=self.retry_policy,
            operation="query_rewrite.invoke",
            on_failure=self._trace_retry_failure,
        )
        result = self._build_result(
            query=query,
            raw_response=coerce_message_content(response.content),
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        self._write_cached_result(result)
        return result

    async def arewrite(self, query: str) -> QueryRewriteResult:
        start = time.perf_counter()
        cached = self._read_cached_result(query, start)
        if cached is not None:
            return cached

        messages = self._build_messages(query)

        async def invoke_model() -> Any:
            if hasattr(self.model, "ainvoke"):
                return await self.model.ainvoke(messages)
            return await asyncio.to_thread(self.model.invoke, messages)

        response = await ainvoke_with_retry(
            invoke_model,
            retry_policy=self.retry_policy,
            operation="query_rewrite.ainvoke",
            on_failure=self._trace_retry_failure,
        )
        result = self._build_result(
            query=query,
            raw_response=coerce_message_content(response.content),
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        self._write_cached_result(result)
        return result

    def _build_messages(self, query: str) -> list[Any]:
        return [
            self.system_prompt,
            HumanMessage(
                content=(
                    "请改写下面这条用户问题，用于商品知识库检索。\n"
                    f"用户问题：{query}"
                )
            ),
        ]

    def _build_result(
        self,
        *,
        query: str,
        raw_response: str,
        elapsed_ms: float,
    ) -> QueryRewriteResult:
        payload = parse_json_object(raw_response)

        rewritten_query = str(payload.get("rewritten_query") or query).strip() or query
        search_queries = self._normalize_queries(payload.get("search_queries"), query)
        if rewritten_query not in search_queries:
            search_queries.insert(0, rewritten_query)
        notes = self._normalize_notes(payload.get("notes"))

        result = QueryRewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            search_queries=search_queries[:3],
            notes=notes,
            raw_response=raw_response,
        )
        if self.trace_recorder is not None:
            self.trace_recorder.event(
                "query_rewrite",
                "query_rewrite.rewrite",
                {
                    "model_name": self.model_name,
                    "provider": self.provider,
                    "original_query": query,
                    "rewritten_query": rewritten_query,
                    "search_queries": search_queries[:3],
                    "notes": notes,
                    "cache_hit": False,
                    "elapsed_ms": elapsed_ms,
                },
            )
        return result

    def _cache_key(self, query: str) -> str | None:
        if self.cache is None:
            return None
        return self.cache.make_key(
            "query_rewrite",
            {
                "query": query,
                "provider": self.provider,
                "model_name": self.model_name,
                "model_kwargs": self.model_kwargs,
                "system_prompt": self.system_prompt.content,
            },
        )

    def _read_cached_result(
        self,
        query: str,
        start: float,
    ) -> QueryRewriteResult | None:
        key = self._cache_key(query)
        if key is None or self.cache is None:
            return None
        payload = self.cache.get_json(key)
        if not isinstance(payload, dict):
            return None

        search_queries = payload.get("search_queries")
        notes = payload.get("notes")
        if not isinstance(search_queries, list):
            return None
        normalized_queries = [str(item) for item in search_queries if str(item)]
        if not normalized_queries:
            return None
        result = QueryRewriteResult(
            original_query=query,
            rewritten_query=str(payload.get("rewritten_query") or query),
            search_queries=normalized_queries,
            notes=(
                [str(item) for item in notes if str(item)]
                if isinstance(notes, list)
                else []
            ),
            raw_response=str(payload.get("raw_response") or ""),
        )
        if self.trace_recorder is not None:
            self.trace_recorder.event(
                "query_rewrite",
                "query_rewrite.rewrite",
                {
                    "model_name": self.model_name,
                    "provider": self.provider,
                    "original_query": query,
                    "rewritten_query": result.rewritten_query,
                    "search_queries": result.search_queries,
                    "notes": result.notes,
                    "cache_hit": True,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
        return result

    def _write_cached_result(self, result: QueryRewriteResult) -> None:
        key = self._cache_key(result.original_query)
        if key is None or self.cache is None:
            return
        self.cache.set_json(
            key,
            {
                "rewritten_query": result.rewritten_query,
                "search_queries": result.search_queries,
                "notes": result.notes,
                "raw_response": result.raw_response,
            },
            ttl_s=self.cache_ttl_s,
        )

    def _trace_retry_failure(self, event: dict[str, Any]) -> None:
        if self.trace_recorder is None:
            return
        self.trace_recorder.event(
            "model",
            "query_rewrite.model_retry",
            {"provider": self.provider, "model_name": self.model_name, **event},
            level="warning" if event.get("will_retry") else "error",
        )

    def _normalize_queries(self, value: Any, original_query: str) -> list[str]:
        if not isinstance(value, list):
            return [original_query]

        queries: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            queries.append(text)

        return queries or [original_query]

    def _normalize_notes(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


def search_with_query_rewrites(
    rag: RAGService,
    original_query: str,
    rewritten_queries: list[str],
    *,
    top_k: int = 3,
    candidate_top_k: int = 10,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    use_bm25: bool | None = None,
    use_rerank: bool | None = None,
    trace_recorder: TraceRecorder | None = None,
) -> list[dict]:
    """Merge candidates from multiple rewritten queries, then rerank by the original query."""
    start = time.perf_counter()
    candidate_map: dict[tuple[str, int], dict] = {}

    for retrieval_query in rewritten_queries:
        results = rag.search_by_hybrid(
            query=retrieval_query,
            top_k=candidate_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            use_bm25=use_bm25,
        )
        for item in results:
            chunk_index = int(item["metadata"].get("chunk_index", -1))
            key = (item["source"], chunk_index)
            existing = candidate_map.get(key)
            if existing is None or item["hybrid_score"] > existing["hybrid_score"]:
                merged = dict(item)
                merged["matched_queries"] = [retrieval_query]
                candidate_map[key] = merged
            elif retrieval_query not in existing["matched_queries"]:
                existing["matched_queries"].append(retrieval_query)

    if not candidate_map:
        if trace_recorder is not None:
            trace_recorder.event(
                "retrieval",
                "query_rewrite.multi_query_search",
                {
                    "original_query": original_query,
                    "rewritten_queries": rewritten_queries,
                    "candidate_count": 0,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
        return []

    merged_candidates = sorted(
        candidate_map.values(),
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )[:candidate_top_k]
    actual_use_rerank = rag.default_use_rerank if use_rerank is None else use_rerank
    if not actual_use_rerank:
        results = merged_candidates[:top_k]
    else:
        results = rag.rerank(original_query, merged_candidates, top_k=top_k)
    if trace_recorder is not None:
        trace_recorder.event(
            "retrieval",
            "query_rewrite.multi_query_search",
            {
                "original_query": original_query,
                "rewritten_queries": rewritten_queries,
                "candidate_count": len(candidate_map),
                "top_k": top_k,
                "candidate_top_k": candidate_top_k,
                "use_rerank": actual_use_rerank,
                "result_count": len(results),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "results": [
                    summarize_result(item, include_content=True) for item in results
                ],
            },
        )
    return results


async def asearch_with_query_rewrites(
    rag: RAGService,
    original_query: str,
    rewritten_queries: list[str],
    *,
    top_k: int = 3,
    candidate_top_k: int = 10,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    use_bm25: bool | None = None,
    use_rerank: bool | None = None,
    trace_recorder: TraceRecorder | None = None,
) -> list[dict]:
    """Async variant that retrieves rewritten queries concurrently."""
    start = time.perf_counter()
    candidate_map: dict[tuple[str, int], dict] = {}

    async def search_one(retrieval_query: str) -> tuple[str, list[dict]]:
        if hasattr(rag, "asearch_by_hybrid"):
            results = await rag.asearch_by_hybrid(
                query=retrieval_query,
                top_k=candidate_top_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                use_bm25=use_bm25,
            )
        elif hasattr(rag, "search_by_hybrid"):
            results = await asyncio.to_thread(
                rag.search_by_hybrid,
                query=retrieval_query,
                top_k=candidate_top_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                use_bm25=use_bm25,
            )
        else:
            results = await asyncio.to_thread(
                rag.search,
                query=retrieval_query,
                top_k=top_k,
                use_bm25=use_bm25,
                use_rerank=False,
                candidate_top_k=candidate_top_k,
            )
        return retrieval_query, results

    query_results = await asyncio.gather(
        *(search_one(retrieval_query) for retrieval_query in rewritten_queries)
    )
    for retrieval_query, results in query_results:
        for item in results:
            chunk_index = int(item["metadata"].get("chunk_index", -1))
            key = (item["source"], chunk_index)
            existing = candidate_map.get(key)
            item_score = float(item.get("hybrid_score", item.get("score", 0.0)))
            existing_score = (
                float(existing.get("hybrid_score", existing.get("score", 0.0)))
                if existing is not None
                else 0.0
            )
            if existing is None or item_score > existing_score:
                merged = dict(item)
                merged["matched_queries"] = [retrieval_query]
                candidate_map[key] = merged
            elif retrieval_query not in existing["matched_queries"]:
                existing["matched_queries"].append(retrieval_query)

    if not candidate_map:
        if trace_recorder is not None:
            trace_recorder.event(
                "retrieval",
                "query_rewrite.multi_query_search",
                {
                    "original_query": original_query,
                    "rewritten_queries": rewritten_queries,
                    "candidate_count": 0,
                    "result_count": 0,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
        return []

    merged_candidates = sorted(
        candidate_map.values(),
        key=lambda item: float(item.get("hybrid_score", item.get("score", 0.0))),
        reverse=True,
    )[:candidate_top_k]
    actual_use_rerank = (
        getattr(rag, "default_use_rerank", False) if use_rerank is None else use_rerank
    )
    if not actual_use_rerank:
        results = merged_candidates[:top_k]
    elif hasattr(rag, "arerank"):
        results = await rag.arerank(original_query, merged_candidates, top_k=top_k)
    else:
        results = await asyncio.to_thread(
            rag.rerank,
            original_query,
            merged_candidates,
            top_k=top_k,
        )
    if trace_recorder is not None:
        trace_recorder.event(
            "retrieval",
            "query_rewrite.multi_query_search",
            {
                "original_query": original_query,
                "rewritten_queries": rewritten_queries,
                "candidate_count": len(candidate_map),
                "top_k": top_k,
                "candidate_top_k": candidate_top_k,
                "use_rerank": actual_use_rerank,
                "result_count": len(results),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "results": [
                    summarize_result(item, include_content=True) for item in results
                ],
            },
        )
    return results
