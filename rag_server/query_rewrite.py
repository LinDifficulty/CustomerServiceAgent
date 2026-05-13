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
from .utils import coerce_message_content, load_prompt, parse_json_object, trace_retry_failure


@dataclass
class QueryRewriteResult:
    """查询改写结果，包含原始查询、改写后查询、多条检索语句和备注信息。"""

    original_query: str  # 用户输入的原始查询
    rewritten_query: str  # LLM 改写后的主查询，更规范、更适合检索
    search_queries: list[str]  # 多条检索语句（最多3条），用于多路融合检索
    notes: list[str]  # 改写过程中 LLM 给出的补充说明
    raw_response: str  # LLM 返回的原始 JSON 文本，便于调试和追踪


class LLMQueryRewriter:
    """使用与 Agent 相同的 LLM 家族来改写检索查询。

    查询改写的目的是将用户口语化、不完整的自然语言问题，转换成更适合
    知识库检索的规范化查询语句，同时保留用户给出的所有硬约束条件。
    支持同步和异步两种调用方式，并内置缓存机制避免重复调用。
    """

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
        # 模型配置：提供商、模型名称、额外参数
        self.provider = provider
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs or {})
        # 如果外部传入了现成的 model 实例则直接使用，否则按配置创建
        self.model = model or create_chat_model(
            provider=provider,
            model_name=model_name,
            **self.model_kwargs,
        )
        # 追踪记录器，用于记录改写过程的各类事件
        self.trace_recorder = trace_recorder
        # 重试策略，控制 LLM 调用失败时的重试行为
        self.retry_policy = retry_policy or LLMRetryPolicy()
        # 缓存服务，用于缓存改写结果，避免相同查询重复调用 LLM
        self.cache = cache
        self.cache_ttl_s = cache_ttl_s  # 缓存有效期（秒），默认 24 小时
        # 系统提示词：定义改写器的角色、约束和输出格式
        self.system_prompt = SystemMessage(content=load_prompt("query_rewrite_system.txt"))

    def _trace_event(self, name: str, payload: dict[str, Any]) -> None:
        """向追踪记录器发送事件（如果配置了 trace_recorder）。"""
        if self.trace_recorder is not None:
            self.trace_recorder.event("query_rewrite", name, payload)

    def rewrite(self, query: str) -> QueryRewriteResult:
        """同步改写查询。

        先检查缓存，命中则直接返回；否则调用 LLM 进行改写，
        并将结果写入缓存供后续复用。
        """
        start = time.perf_counter()  # 记录开始时间，用于计算耗时
        # 1. 先尝试从缓存读取改写结果
        cached = self._read_cached_result(query, start)
        if cached is not None:
            return cached

        # 2. 缓存未命中，构建消息并调用 LLM
        messages = self._build_messages(query)
        # 使用带重试机制的调用，失败时自动重试
        response = invoke_with_retry(
            lambda: self.model.invoke(messages),
            retry_policy=self.retry_policy,
            operation="query_rewrite.invoke",
            on_failure=self._trace_retry_failure,
        )
        # 3. 解析 LLM 返回的 JSON，构建结果对象
        result = self._build_result(
            query=query,
            raw_response=coerce_message_content(response.content),
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        # 4. 将结果写入缓存
        self._write_cached_result(result)
        return result

    async def arewrite(self, query: str) -> QueryRewriteResult:
        """异步改写查询。

        与同步版本逻辑相同，但 LLM 调用使用异步方式，
        如果模型本身不支持异步则回退到线程池中同步执行。
        """
        start = time.perf_counter()
        # 1. 先尝试从缓存读取改写结果
        cached = self._read_cached_result(query, start)
        if cached is not None:
            return cached

        # 2. 构建消息
        messages = self._build_messages(query)

        # 3. 定义一个异步调用函数，兼容同步模型
        async def invoke_model() -> Any:
            if hasattr(self.model, "ainvoke"):  # 模型支持原生异步调用
                return await self.model.ainvoke(messages)
            # 模型不支持异步，放到线程池中执行以避免阻塞事件循环
            return await asyncio.to_thread(self.model.invoke, messages)

        # 使用异步带重试机制的调用
        response = await ainvoke_with_retry(
            invoke_model,
            retry_policy=self.retry_policy,
            operation="query_rewrite.ainvoke",
            on_failure=self._trace_retry_failure,
        )
        # 4. 解析结果并缓存
        result = self._build_result(
            query=query,
            raw_response=coerce_message_content(response.content),
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        self._write_cached_result(result)
        return result

    def _build_messages(self, query: str) -> list[Any]:
        """构建发送给 LLM 的消息列表（系统提示词 + 用户问题）。"""
        return [
            self.system_prompt,
            HumanMessage(content=(f"请改写下面这条用户问题，用于商品知识库检索。\n用户问题：{query}")),
        ]

    def _build_result(
        self,
        *,
        query: str,
        raw_response: str,
        elapsed_ms: float,
    ) -> QueryRewriteResult:
        """从 LLM 的原始响应中解析出结构化的改写结果。

        会解析 JSON、规范化查询列表、确保改写后的查询排在首位，
        并记录追踪事件。
        """
        # 从原始 JSON 响应中提取字段
        payload = parse_json_object(raw_response)

        # 提取改写后查询，如果解析失败或为空则回退到原始查询
        rewritten_query = str(payload.get("rewritten_query") or query).strip() or query
        # 规范化检索语句列表
        search_queries = self._normalize_queries(payload.get("search_queries"), query)
        # 确保改写后的主查询排在检索语句列表的第一位
        if rewritten_query not in search_queries:
            search_queries.insert(0, rewritten_query)
        # 规范化备注列表
        notes = self._normalize_notes(payload.get("notes"))

        # 构建结果对象，最多保留 3 条检索语句
        result = QueryRewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            search_queries=search_queries[:3],
            notes=notes,
            raw_response=raw_response,
        )
        self._trace_event(
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
        """生成缓存的键，基于查询内容和模型配置。

        如果未配置缓存服务则返回 None。键中包含查询文本、模型信息
        和系统提示词，确保不同配置下的改写结果不会互相污染。
        """
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
        """从缓存读取改写结果。

        校验缓存数据完整性后才返回，避免脏数据影响检索质量。
        如果在缓存中找到有效结果，同时会记录一条带 cache_hit=True 的追踪事件。
        """
        key = self._cache_key(query)
        if key is None or self.cache is None:
            return None
        payload = self.cache.get_json(key)
        # 缓存数据必须是字典类型
        if not isinstance(payload, dict):
            return None

        # 校验检索语句列表的完整性
        search_queries = payload.get("search_queries")
        notes = payload.get("notes")
        if not isinstance(search_queries, list):
            return None
        normalized_queries = [str(item) for item in search_queries if str(item)]
        if not normalized_queries:
            return None
        # 构建结果对象
        result = QueryRewriteResult(
            original_query=query,
            rewritten_query=str(payload.get("rewritten_query") or query),
            search_queries=normalized_queries,
            notes=([str(item) for item in notes if str(item)] if isinstance(notes, list) else []),
            raw_response=str(payload.get("raw_response") or ""),
        )
        self._trace_event(
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
        """将改写结果写入缓存，供后续相同查询复用。"""
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
        """记录 LLM 调用重试失败的事件。"""
        trace_retry_failure(
            self.trace_recorder,
            "model",
            "query_rewrite.model_retry",
            self.provider,
            self.model_name,
            event,
        )

    def _normalize_queries(self, value: Any, original_query: str) -> list[str]:
        """规范化检索语句列表：去重、去空、维护顺序。

        如果输入不是列表或处理后列表为空，则回退到只包含原始查询。
        """
        if not isinstance(value, list):
            return [original_query]

        queries: list[str] = []
        seen: set[str] = set()  # 用于去重
        for item in value:
            text = str(item).strip()
            if not text or text in seen:  # 跳过空字符串和重复项
                continue
            seen.add(text)
            queries.append(text)

        return queries or [original_query]  # 最终回退：至少返回原始查询

    def _normalize_notes(self, value: Any) -> list[str]:
        """规范化备注列表：去空字符串，保留有效备注。"""
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


def _merge_search_result(
    item: dict,
    retrieval_query: str,
    candidate_map: dict[tuple[str, int], dict],
) -> None:
    """Merge a search result item into the candidate map, keeping highest score per key."""
    chunk_index = int(item["metadata"].get("chunk_index", -1))
    key = (item["source"], chunk_index)
    existing = candidate_map.get(key)
    item_score = float(item.get("hybrid_score", item.get("score", 0.0)))
    existing_score = float(existing.get("hybrid_score", existing.get("score", 0.0))) if existing is not None else 0.0
    if existing is None or item_score > existing_score:
        merged = dict(item)
        merged["matched_queries"] = [retrieval_query]
        candidate_map[key] = merged
    elif retrieval_query not in existing["matched_queries"]:
        existing["matched_queries"].append(retrieval_query)


async def _multi_query_search_impl(
    rag: RAGService,
    original_query: str,
    rewritten_queries: list[str],
    *,
    top_k: int,
    candidate_top_k: int,
    vector_weight: float,
    bm25_weight: float,
    use_bm25: bool | None,
    use_rerank: bool | None,
    trace_recorder: TraceRecorder | None,
    use_async: bool,
) -> list[dict]:
    """Shared implementation for multi-query rewrite search (sync and async)."""
    start = time.perf_counter()
    candidate_map: dict[tuple[str, int], dict] = {}

    if use_async:

        async def _search_one_async(retrieval_query: str) -> tuple[str, list[dict]]:
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

        query_results = await asyncio.gather(*(_search_one_async(q) for q in rewritten_queries))
        for retrieval_query, results in query_results:
            for item in results:
                _merge_search_result(item, retrieval_query, candidate_map)
    else:
        for retrieval_query in rewritten_queries:
            results = rag.search_by_hybrid(
                query=retrieval_query,
                top_k=candidate_top_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                use_bm25=use_bm25,
            )
            for item in results:
                _merge_search_result(item, retrieval_query, candidate_map)

    # Empty result trace and early return
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

    # Sort and deduplicate candidates
    merged_candidates = sorted(
        candidate_map.values(),
        key=lambda item: float(item.get("hybrid_score", item.get("score", 0.0))),
        reverse=True,
    )[:candidate_top_k]

    # Rerank or truncate
    actual_use_rerank = getattr(rag, "default_use_rerank", False) if use_rerank is None else use_rerank
    if not actual_use_rerank:
        results = merged_candidates[:top_k]
    elif use_async and hasattr(rag, "arerank"):
        results = await rag.arerank(original_query, merged_candidates, top_k=top_k)
    elif use_async:
        results = await asyncio.to_thread(rag.rerank, original_query, merged_candidates, top_k=top_k)
    else:
        results = rag.rerank(original_query, merged_candidates, top_k=top_k)

    # Trace event
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
                "results": [summarize_result(item, include_content=True) for item in results],
            },
        )
    return results


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
    """多条改写查询合并检索：分别检索，合并去重候选项，再用原查询重排序。"""
    return asyncio.run(
        _multi_query_search_impl(
            rag,
            original_query,
            rewritten_queries,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            use_bm25=use_bm25,
            use_rerank=use_rerank,
            trace_recorder=trace_recorder,
            use_async=False,
        )
    )


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
    """多条改写查询的异步合并检索（并发执行各改写查询，减少总延迟）。"""
    return await _multi_query_search_impl(
        rag,
        original_query,
        rewritten_queries,
        top_k=top_k,
        candidate_top_k=candidate_top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        use_bm25=use_bm25,
        use_rerank=use_rerank,
        trace_recorder=trace_recorder,
        use_async=True,
    )
