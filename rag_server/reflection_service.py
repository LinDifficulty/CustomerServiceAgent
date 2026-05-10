from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_retry import LLMRetryPolicy, ainvoke_with_retry
from .model_factory import DEFAULT_CHAT_MODEL, DEFAULT_CHAT_PROVIDER, create_chat_model
from .rag_service import RAGService
from .trace_service import TraceRecorder, preview_text, summarize_result
from .utils import coerce_bool, coerce_message_content, parse_json_object

# 默认使用与 Agent 对话模型相同的模型进行反思审校
DEFAULT_REFLECTION_MODEL = DEFAULT_CHAT_MODEL


@dataclass(frozen=True)
class ReflectionResult:
    """反思审校的结构化判断结果，由反思模型生成。

    包含幻觉判断、证据充分性评估、修正建议等关键字段，
    用于驱动后续的补充检索和回答修正流程。
    """

    has_hallucination: bool       # 初次回答是否包含未被证据支持的幻觉内容
    needs_more_evidence: bool     # 是否需要补充更多证据来支撑回答
    reason: str                   # 一句话说明审校结论的原因
    search_query: str             # 如果需要补检索，此字段包含精简的检索问题
    correction_guidance: str      # 如果需要修正，说明应该删除或修改什么内容
    raw_response: str             # LLM 返回的原始文本，便于调试追踪

    @property
    def needs_revision(self) -> bool:
        """回答是否需要修正：存在幻觉 或 需要补充证据时都需要修正。"""
        return self.has_hallucination or self.needs_more_evidence


class ReflectionAgent:
    """反思审校 Agent：审查初次回答质量，必要时补充证据并修正回答。

    工作流程：
    1. reflect：审校初次回答，判断是否有幻觉或证据不足
    2. 如果不需要修正，直接返回原始回答
    3. 如果需要修正，进行补充检索获取更多证据
    4. revise：基于原有证据和补充证据，修正回答
    5. 如果任一环节失败，安全回退到原始回答
    """

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
        # RAG 检索服务，用于补充检索
        self.rag = rag
        # 模型配置
        self.provider = provider
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs or {})
        self.model = model or create_chat_model(
            provider=provider,
            model_name=model_name,
            **self.model_kwargs,
        )
        # 重试策略：LLM 调用失败时的重试控制
        self.retry_policy = retry_policy or LLMRetryPolicy()
        # 追踪记录器
        self.trace_recorder = trace_recorder
        # 补充检索参数：检索多少条和候选多少条
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
        """主流程：审校初次回答，必要时修正并返回最终回答。

        整个流程为异步，任何步骤失败都会安全回退到原始回答，
        不会因反思本身的问题而中断用户请求。
        """
        # 空输入保护：如果问题或回答为空，直接返回原始回答
        if not user_question.strip() or not initial_answer.strip():
            return initial_answer

        # 步骤 1：审校初次回答
        try:
            reflection = await self.reflect(
                user_question=user_question,
                initial_answer=initial_answer,
                evidence_context=evidence_context,
                memory_context=memory_context,
                skill_context=skill_context,
            )
        except Exception as error:
            # 审校失败时记录告警并安全回退
            self._trace_event(
                "reflection.reflect_failed",
                {"error": repr(error), "question": user_question},
                level="warning",
            )
            return initial_answer

        # 步骤 2：如果不需要修正，直接返回原始回答
        if not reflection.needs_revision:
            return initial_answer

        # 步骤 3：需要修正，先进行补充检索获取更多证据
        supplemental_context = self._supplemental_retrieval(
            reflection.search_query or user_question
        )
        # 如果补充检索也没有获取到任何证据，记录告警并回退
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

        # 步骤 4：基于证据修正回答
        try:
            return await self.revise(
                user_question=user_question,
                initial_answer=initial_answer,
                reflection=reflection,
                evidence_context=evidence_context,
                supplemental_context=supplemental_context,
            )
        except Exception as error:
            # 修正失败时记录告警并安全回退
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
        """审校初次回答：调用 LLM 判断是否存在幻觉或证据不足。

        通过精心设计的提示词，让 LLM 以严格的客服审校角色审视回答，
        区分证据（商品事实）和记忆上下文（用户偏好），
        避免将用户偏好误认为事实依据。
        """
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
        # 使用带重试机制的异步 LLM 调用
        response = await ainvoke_with_retry(
            lambda: self.model.ainvoke(messages),
            retry_policy=self.retry_policy,
            operation="reflection.reflect",
            on_failure=self._trace_retry_failure,
        )
        # 从 LLM 返回的 JSON 中解析结构化审校结果
        raw_response = coerce_message_content(response.content)
        result = parse_reflection_result(raw_response)
        # 记录审校完成追踪事件
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
        """基于审校结论和补充证据修正回答。

        将审校结论（是否有幻觉、缺少什么证据、修正建议）和补充检索到的证据
        一起提供给 LLM，让它生成修正后的最终回复。
        要求 LLM 以客服身份直接输出，不暴露内部审校过程。
        """
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
        # 带重试的异步 LLM 调用
        response = await ainvoke_with_retry(
            lambda: self.model.ainvoke(messages),
            retry_policy=self.retry_policy,
            operation="reflection.revise",
            on_failure=self._trace_retry_failure,
        )
        revised_answer = coerce_message_content(response.content).strip()
        # 记录修正完成追踪事件
        self._trace_event(
            "reflection.revise",
            {
                "initial_preview": preview_text(initial_answer),
                "revised_preview": preview_text(revised_answer),
            },
        )
        # 如果修正后的回答为空，回退到原始回答
        return revised_answer or initial_answer

    def _supplemental_retrieval(self, query: str) -> str:
        """根据审校推荐的检索问题，从知识库中补充检索证据。

        如果检索失败或返回空结果，记录告警并返回空字符串，
        不会中断主流程。
        """
        if not query.strip():
            return ""
        try:
            # 使用基础检索方法（不使用 BM25 和 rerank，保持快速）
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

        # 将检索结果格式化为可读的文本上下文
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
        """记录 LLM 调用重试失败事件。

        根据是否会继续重试使用不同的日志级别（warning/error）。
        """
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
        """统一的追踪事件记录方法。

        封装了对 trace_recorder 的判空逻辑，避免重复的 None 检查。
        """
        if self.trace_recorder is None:
            return
        self.trace_recorder.event("reflection", name, payload, level=level)


def parse_reflection_result(raw_response: str) -> ReflectionResult:
    """从 LLM 返回的原始 JSON 字符串中解析出结构化的审校结果。

    使用 parse_json_object 安全解析 JSON，给每个字段提供默认值，
    避免因 LLM 输出格式不规范导致解析失败。
    """
    payload = parse_json_object(raw_response)
    return ReflectionResult(
        has_hallucination=coerce_bool(payload.get("has_hallucination")),
        needs_more_evidence=coerce_bool(payload.get("needs_more_evidence")),
        reason=str(payload.get("reason") or ""),
        search_query=str(payload.get("search_query") or ""),
        correction_guidance=str(payload.get("correction_guidance") or ""),
        raw_response=raw_response,
    )


def format_retrieval_results(results: list[dict]) -> str:
    """将检索结果列表格式化为结构化的文本块，用于作为 LLM 提示词的上下文。

    每个结果包含序号、来源文件名和内容，结果之间用双换行分隔。
    """
    if not results:
        return ""

    blocks: list[str] = []
    # 从 1 开始编号，便于 LLM 理解和引用
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
