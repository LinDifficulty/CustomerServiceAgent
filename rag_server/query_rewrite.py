from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

from .llm_retry import LLMRetryPolicy, invoke_with_retry
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
        model_name: str = "qwen3-max-2026-01-23",
        model: ChatTongyi | None = None,
        trace_recorder: TraceRecorder | None = None,
        retry_policy: LLMRetryPolicy | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model or ChatTongyi(model=model_name, max_retries=0)
        self.trace_recorder = trace_recorder
        self.retry_policy = retry_policy or LLMRetryPolicy()
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
        messages = [
            self.system_prompt,
            HumanMessage(
                content=(
                    "请改写下面这条用户问题，用于商品知识库检索。\n"
                    f"用户问题：{query}"
                )
            ),
        ]
        response = invoke_with_retry(
            lambda: self.model.invoke(messages),
            retry_policy=self.retry_policy,
            operation="query_rewrite.invoke",
            on_failure=self._trace_retry_failure,
        )
        raw_response = coerce_message_content(response.content)
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
                    "original_query": query,
                    "rewritten_query": rewritten_query,
                    "search_queries": search_queries[:3],
                    "notes": notes,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                },
            )
        return result

    def _trace_retry_failure(self, event: dict[str, Any]) -> None:
        if self.trace_recorder is None:
            return
        self.trace_recorder.event(
            "model",
            "query_rewrite.model_retry",
            {"model_name": self.model_name, **event},
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
