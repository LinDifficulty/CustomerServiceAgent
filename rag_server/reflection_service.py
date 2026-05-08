from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_retry import LLMRetryPolicy, ainvoke_with_retry
from .model_factory import DEFAULT_CHAT_MODEL, DEFAULT_CHAT_PROVIDER, create_chat_model
from .rag_service import RAGService
from .trace_service import TraceRecorder, preview_text, summarize_result
from .utils import coerce_bool, coerce_message_content, parse_json_object

DEFAULT_REFLECTION_MODEL = DEFAULT_CHAT_MODEL


@dataclass(frozen=True)
class ReflectionResult:
    """Structured judgment produced by the reflection model."""

    has_hallucination: bool
    needs_more_evidence: bool
    reason: str
    search_query: str
    correction_guidance: str
    raw_response: str

    @property
    def needs_revision(self) -> bool:
        return self.has_hallucination or self.needs_more_evidence


class ReflectionAgent:
    """Review an answer, retrieve more evidence if needed, then revise it."""

    def __init__(
        self,
        *,
        rag: RAGService,
        model_name: str = DEFAULT_REFLECTION_MODEL,
        provider: str = DEFAULT_CHAT_PROVIDER,
        model_kwargs: dict[str, Any] | None = None,
        model: Any | None = None,
        retry_policy: LLMRetryPolicy | None = None,
        trace_recorder: TraceRecorder | None = None,
        supplemental_top_k: int = 3,
        supplemental_candidate_top_k: int = 10,
    ) -> None:
        self.rag = rag
        self.provider = provider
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs or {})
        self.model = model or create_chat_model(
            provider=provider,
            model_name=model_name,
            **self.model_kwargs,
        )
        self.retry_policy = retry_policy or LLMRetryPolicy()
        self.trace_recorder = trace_recorder
        self.supplemental_top_k = supplemental_top_k
        self.supplemental_candidate_top_k = supplemental_candidate_top_k

    async def review_and_revise(
        self,
        *,
        user_question: str,
        initial_answer: str,
        evidence_context: str = "",
        memory_context: str = "",
        skill_context: str = "",
    ) -> str:
        """Run reflection; return the original or revised answer."""
        if not user_question.strip() or not initial_answer.strip():
            return initial_answer

        try:
            reflection = await self.reflect(
                user_question=user_question,
                initial_answer=initial_answer,
                evidence_context=evidence_context,
                memory_context=memory_context,
                skill_context=skill_context,
            )
        except Exception as error:
            self._trace_event(
                "reflection.reflect_failed",
                {"error": repr(error), "question": user_question},
                level="warning",
            )
            return initial_answer

        if not reflection.needs_revision:
            return initial_answer

        supplemental_context = self._supplemental_retrieval(
            reflection.search_query or user_question
        )
        if not supplemental_context and not evidence_context:
            self._trace_event(
                "reflection.no_evidence_for_revision",
                {
                    "reason": reflection.reason,
                    "search_query": reflection.search_query,
                },
                level="warning",
            )
            return initial_answer

        try:
            return await self.revise(
                user_question=user_question,
                initial_answer=initial_answer,
                reflection=reflection,
                evidence_context=evidence_context,
                supplemental_context=supplemental_context,
            )
        except Exception as error:
            self._trace_event(
                "reflection.revision_failed",
                {"error": repr(error), "question": user_question},
                level="warning",
            )
            return initial_answer

    async def reflect(
        self,
        *,
        user_question: str,
        initial_answer: str,
        evidence_context: str = "",
        memory_context: str = "",
        skill_context: str = "",
    ) -> ReflectionResult:
        messages = [
            SystemMessage(
                content=(
                    "你是一个严格的客服回答审校器。你的任务是判断初次回答是否包含"
                    "没有被证据支持的商品事实、政策承诺、库存参数、尺码建议或售后规则。"
                    "只输出 JSON，不要输出解释性正文。"
                )
            ),
            HumanMessage(
                content=(
                    "请审查下面回答是否可能 hallucination。\n\n"
                    f"用户问题:\n{user_question}\n\n"
                    f"已有工具或检索证据:\n{evidence_context or '无'}\n\n"
                    f"长期记忆上下文（只能代表用户偏好，不是商品事实）:\n"
                    f"{memory_context or '无'}\n\n"
                    f"Skill 上下文:\n{skill_context or '无'}\n\n"
                    f"初次回答:\n{initial_answer}\n\n"
                    "输出 JSON，字段如下：\n"
                    "{\n"
                    '  "has_hallucination": true/false,\n'
                    '  "needs_more_evidence": true/false,\n'
                    '  "reason": "一句话说明",\n'
                    '  "search_query": "如果需要补检索，给出精简检索问题，否则为空字符串",\n'
                    '  "correction_guidance": "如果需要修正，说明应删改什么，否则为空字符串"\n'
                    "}"
                )
            ),
        ]
        response = await ainvoke_with_retry(
            lambda: self.model.ainvoke(messages),
            retry_policy=self.retry_policy,
            operation="reflection.reflect",
            on_failure=self._trace_retry_failure,
        )
        raw_response = coerce_message_content(response.content)
        result = parse_reflection_result(raw_response)
        self._trace_event(
            "reflection.reflect",
            {
                "has_hallucination": result.has_hallucination,
                "needs_more_evidence": result.needs_more_evidence,
                "reason": result.reason,
                "search_query": result.search_query,
                "correction_guidance": result.correction_guidance,
                "raw_preview": preview_text(raw_response),
            },
        )
        return result

    async def revise(
        self,
        *,
        user_question: str,
        initial_answer: str,
        reflection: ReflectionResult,
        evidence_context: str,
        supplemental_context: str,
    ) -> str:
        messages = [
            SystemMessage(
                content=(
                    "你是一个客服回答修正器。请基于证据修正回答，删除没有证据支持的"
                    "具体事实或承诺；如果证据仍不足，就明确说当前无法确认。"
                    "不要提到 reflection、审校过程或内部检索流程。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题:\n{user_question}\n\n"
                    f"初次回答:\n{initial_answer}\n\n"
                    "审校结论:\n"
                    f"- 是否疑似 hallucination: {reflection.has_hallucination}\n"
                    f"- 是否需要更多证据: {reflection.needs_more_evidence}\n"
                    f"- 原因: {reflection.reason}\n"
                    f"- 修正建议: {reflection.correction_guidance}\n\n"
                    f"已有证据:\n{evidence_context or '无'}\n\n"
                    f"补充检索证据:\n{supplemental_context or '无'}\n\n"
                    "请给出最终客服回复。"
                )
            ),
        ]
        response = await ainvoke_with_retry(
            lambda: self.model.ainvoke(messages),
            retry_policy=self.retry_policy,
            operation="reflection.revise",
            on_failure=self._trace_retry_failure,
        )
        revised_answer = coerce_message_content(response.content).strip()
        self._trace_event(
            "reflection.revise",
            {
                "initial_preview": preview_text(initial_answer),
                "revised_preview": preview_text(revised_answer),
            },
        )
        return revised_answer or initial_answer

    def _supplemental_retrieval(self, query: str) -> str:
        if not query.strip():
            return ""
        try:
            results = self.rag.search(
                query=query,
                top_k=self.supplemental_top_k,
                candidate_top_k=self.supplemental_candidate_top_k,
            )
        except Exception as error:
            self._trace_event(
                "reflection.supplemental_retrieval_failed",
                {"query": query, "error": repr(error)},
                level="warning",
            )
            return ""

        context = format_retrieval_results(results)
        self._trace_event(
            "reflection.supplemental_retrieval",
            {
                "query": query,
                "result_count": len(results),
                "results": [
                    summarize_result(item, include_content=True)
                    for item in results
                ],
            },
        )
        return context

    def _trace_retry_failure(self, event: dict[str, Any]) -> None:
        self._trace_event(
            "reflection.model_retry",
            {"provider": self.provider, "model_name": self.model_name, **event},
            level="warning" if event.get("will_retry") else "error",
        )

    def _trace_event(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        level: str = "info",
    ) -> None:
        if self.trace_recorder is None:
            return
        self.trace_recorder.event("reflection", name, payload, level=level)


def parse_reflection_result(raw_response: str) -> ReflectionResult:
    payload = parse_json_object(raw_response)
    return ReflectionResult(
        has_hallucination=_coerce_bool(payload.get("has_hallucination")),
        needs_more_evidence=_coerce_bool(payload.get("needs_more_evidence")),
        reason=str(payload.get("reason") or ""),
        search_query=str(payload.get("search_query") or ""),
        correction_guidance=str(payload.get("correction_guidance") or ""),
        raw_response=raw_response,
    )


def format_retrieval_results(results: list[dict]) -> str:
    if not results:
        return ""

    blocks: list[str] = []
    for index, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"片段{index}",
                    f"来源: {item.get('source', '')}",
                    f"内容: {item.get('content', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _coerce_bool(value: Any) -> bool:
    return coerce_bool(value)
