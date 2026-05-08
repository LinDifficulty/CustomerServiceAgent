from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import (
    DEFAULT_CLI_CONFIG_OUTPUT_ENABLED,
    DEFAULT_LIVE_EVENTS_ENABLED,
    ConfigError,
    load_app_config,
)
from .llm_retry import LLMRetryError, LLMRetryPolicy, ainvoke_with_retry
from .memory_service import LLMMemoryExtractor, MemoryService
from .mcp_service import DEFAULT_MCP_CONFIG_PATH, load_mcp_tools_from_config
from .model_factory import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_PROVIDER,
    create_chat_model,
)
from .query_rewrite import LLMQueryRewriter, search_with_query_rewrites
from .rag_service import RAGService
from .reflection_service import ReflectionAgent
from .skill_service import SkillRegistry, build_skill_tools
from .trace_service import DEFAULT_TRACE_DIR, TraceRecorder, preview_text
from .utils import coerce_message_content

DEFAULT_AGENT_MODEL = DEFAULT_CHAT_MODEL
DEFAULT_USER_ID = "default_user"
DEFAULT_QUERY_REWRITE_MODE = "on"
QUERY_REWRITE_MODES = ("on", "off", "rewrite_only", "multi_query")
DEFAULT_MAX_TOOL_ROUNDS = 6
DEFAULT_MAX_REPEATED_TOOL_CALLS = 2
DEFAULT_REFLECTION_ENABLED = True


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    latest_user_message: str
    memory_context: str
    skill_context: str
    active_skill_names: list[str]
    tool_round_count: int
    last_tool_call_signature: str
    repeated_tool_call_count: int


def normalize_query_rewrite_mode(mode: str) -> str:
    """Convert the user-facing on/off switch into the internal strategy name."""
    return "multi_query" if mode == "on" else mode


def format_tongyi_error(
    model: Any,
    messages: list[BaseMessage],
    error: Exception,
) -> str:
    """把模型异常转换成更可读的 CLI 错误信息。"""
    if isinstance(error, LLMRetryError):
        prefix = (
            "大模型多次尝试后仍未及时响应或暂时不可用。"
            if error.attempts > 1
            else "大模型调用未及时响应或暂时不可用。"
        )
        return (
            f"{prefix} attempts={error.attempts}, "
            f"last_error={error.last_error!r}"
        )

    status_code = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    message = getattr(error, "message", None)
    request_id = getattr(error, "request_id", None)
    if status_code and code and message:
        return (
            "大模型调用失败。"
            f" status_code={status_code}, code={code}, message={message},"
            f" request_id={request_id}"
        )

    return f"大模型调用失败：{error!r}"


def build_retrieval_tool(
    rag: RAGService,
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    rewrite_provider: str = DEFAULT_CHAT_PROVIDER,
    rewrite_model_name: str = DEFAULT_AGENT_MODEL,
    rewrite_model_kwargs: dict[str, Any] | None = None,
    llm_retry_policy: LLMRetryPolicy | None = None,
    trace_recorder: TraceRecorder | None = None,
):
    actual_mode = normalize_query_rewrite_mode(query_rewrite_mode)
    rewriter = (
        LLMQueryRewriter(
            provider=rewrite_provider,
            model_name=rewrite_model_name,
            model_kwargs=rewrite_model_kwargs,
            trace_recorder=trace_recorder,
            retry_policy=llm_retry_policy,
        )
        if actual_mode != "off"
        else None
    )
    return build_retrieval_tool_with_rewrite(
        rag,
        query_rewrite_mode=actual_mode,
        rewriter=rewriter,
        trace_recorder=trace_recorder,
    )


def build_retrieval_tool_with_rewrite(
    rag: RAGService,
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    rewriter: LLMQueryRewriter | None = None,
    trace_recorder: TraceRecorder | None = None,
):
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)

    @tool(description="检索商品知识，返回与用户问题最相关的商品信息片段。")
    def search_product_knowledge(question: str) -> str:
        """检索商品知识库，返回与用户问题最相关的商品信息片段。"""
        trace_payload: dict[str, Any] = {
            "question": question,
            "query_rewrite_mode": actual_query_rewrite_mode,
        }
        if actual_query_rewrite_mode == "rewrite_only" and rewriter is not None:
            try:
                rewrite_result = rewriter.rewrite(question)
            except Exception as error:
                trace_payload["query_rewrite_error"] = repr(error)
                trace_payload["retrieval_queries"] = [question]
                results = rag.search(
                    query=question,
                    top_k=3,
                    candidate_top_k=10,
                )
                if trace_recorder is not None:
                    trace_recorder.event(
                        "query_rewrite",
                        "query_rewrite.fallback_to_original",
                        trace_payload,
                        level="warning",
                    )
            else:
                trace_payload["rewritten_query"] = rewrite_result.rewritten_query
                trace_payload["retrieval_queries"] = [rewrite_result.rewritten_query]
                results = rag.search(
                    query=rewrite_result.rewritten_query,
                    top_k=3,
                    candidate_top_k=10,
                )
        elif actual_query_rewrite_mode == "multi_query" and rewriter is not None:
            try:
                rewrite_result = rewriter.rewrite(question)
            except Exception as error:
                trace_payload["query_rewrite_error"] = repr(error)
                trace_payload["retrieval_queries"] = [question]
                results = rag.search(
                    query=question,
                    top_k=3,
                    candidate_top_k=10,
                )
                if trace_recorder is not None:
                    trace_recorder.event(
                        "query_rewrite",
                        "query_rewrite.fallback_to_original",
                        trace_payload,
                        level="warning",
                    )
            else:
                retrieval_queries = [question, *rewrite_result.search_queries]
                deduplicated_queries = list(dict.fromkeys(retrieval_queries))
                trace_payload["rewritten_query"] = rewrite_result.rewritten_query
                trace_payload["retrieval_queries"] = deduplicated_queries
                results = search_with_query_rewrites(
                    rag,
                    original_query=question,
                    rewritten_queries=deduplicated_queries,
                    top_k=3,
                    candidate_top_k=10,
                    trace_recorder=trace_recorder,
                )
        else:
            trace_payload["retrieval_queries"] = [question]
            results = rag.search(
                query=question,
                top_k=3,
                candidate_top_k=10,
            )

        if not results:
            if trace_recorder is not None:
                trace_recorder.event(
                    "tool",
                    "tool.search_product_knowledge",
                    {**trace_payload, "result_count": 0},
                )
            return "未检索到相关商品知识。"

        blocks: list[str] = []
        for index, item in enumerate(results, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"片段{index}",
                        f"来源: {item['source']}",
                        f"内容: {item['content']}",
                    ]
                )
            )
        output = "\n\n".join(blocks)
        if trace_recorder is not None:
            trace_recorder.event(
                "tool",
                "tool.search_product_knowledge",
                {
                    **trace_payload,
                    "result_count": len(results),
                    "output_preview": preview_text(output),
                },
            )
        return output

    return search_product_knowledge


def latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return coerce_message_content(message.content).strip()
    return ""


def latest_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def format_recent_tool_context(
    messages: list[BaseMessage],
    *,
    max_messages: int = 6,
    max_chars: int = 6000,
) -> str:
    tool_messages = [
        message for message in messages if isinstance(message, ToolMessage)
    ][-max_messages:]
    if not tool_messages:
        return ""

    blocks = []
    for index, message in enumerate(tool_messages, start=1):
        content = coerce_message_content(message.content).strip()
        if not content:
            continue
        blocks.append(f"工具结果{index}:\n{content}")

    context = "\n\n".join(blocks)
    if len(context) <= max_chars:
        return context
    return context[-max_chars:]


def replace_ai_message_content(message: AIMessage, content: str) -> AIMessage:
    try:
        return message.model_copy(update={"content": content})
    except AttributeError:
        try:
            return message.copy(update={"content": content})
        except Exception:
            return AIMessage(content=content)


def message_usage_metadata(message: BaseMessage) -> dict[str, Any]:
    """Extract provider token usage without depending on one provider schema."""
    usage: dict[str, Any] = {}
    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        usage["usage_metadata"] = usage_metadata

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if token_usage is not None:
            usage["token_usage"] = token_usage
        model_name = response_metadata.get("model_name") or response_metadata.get(
            "model"
        )
        if model_name is not None:
            usage["model_name"] = model_name
        finish_reason = response_metadata.get("finish_reason")
        if finish_reason is not None:
            usage["finish_reason"] = finish_reason
    return usage


def _compact_live_value(value: Any, *, max_chars: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, bool | int | float):
        text = str(value)
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return preview_text(text.replace("\n", "\\n"), max_chars=max_chars)


def _live_elapsed_ms(record: dict[str, Any], payload: dict[str, Any]) -> float | None:
    elapsed = record.get("elapsed_ms")
    if not isinstance(elapsed, int | float):
        elapsed = payload.get("elapsed_ms")
    return float(elapsed) if isinstance(elapsed, int | float) else None


def _format_live_elapsed(record: dict[str, Any], payload: dict[str, Any]) -> str:
    elapsed = _live_elapsed_ms(record, payload)
    if elapsed is None:
        return ""
    return f" elapsed={elapsed:.1f}ms"


def _format_live_level(record: dict[str, Any]) -> str:
    level = str(record.get("level") or "info")
    return "" if level == "info" else f" level={level}"


def _format_rag_live_event(
    name: str,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> str:
    query = (
        payload.get("query")
        or payload.get("question")
        or payload.get("original_query")
    )
    parts = [f"[实时][RAG] {name}"]
    if query:
        parts.append(f'query="{_compact_live_value(query, max_chars=90)}"')
    if payload.get("rewritten_query"):
        parts.append(
            f'rewrite="{_compact_live_value(payload["rewritten_query"], max_chars=90)}"'
        )
    if payload.get("retrieval_queries"):
        parts.append(
            "queries="
            + _compact_live_value(payload["retrieval_queries"], max_chars=140)
        )
    if payload.get("candidate_count") is not None:
        parts.append(f"candidates={payload['candidate_count']}")
    if payload.get("result_count") is not None:
        parts.append(f"results={payload['result_count']}")
    if payload.get("use_rerank") is not None:
        parts.append(f"rerank={payload['use_rerank']}")
    return " ".join(parts) + _format_live_level(record) + _format_live_elapsed(
        record,
        payload,
    )


def _format_memory_live_event(
    name: str,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> str:
    parts = [f"[实时][Memory] {name}"]
    if payload.get("user_id"):
        parts.append(f"user={payload['user_id']}")
    if payload.get("query"):
        parts.append(f'query="{_compact_live_value(payload["query"], max_chars=90)}"')
    layer_counts = payload.get("layer_counts")
    if isinstance(layer_counts, dict):
        parts.append(
            "layers="
            + ",".join(
                f"{layer}:{count}" for layer, count in sorted(layer_counts.items())
            )
        )
    if payload.get("memory_types"):
        parts.append("types=" + _compact_live_value(payload["memory_types"]))
    if payload.get("result_count") is not None:
        parts.append(f"results={payload['result_count']}")
    if payload.get("new_memory_count") is not None:
        parts.append(f"new={payload['new_memory_count']}")
    if payload.get("reason"):
        parts.append(f"reason={payload['reason']}")
    return " ".join(parts) + _format_live_level(record) + _format_live_elapsed(
        record,
        payload,
    )


def _format_skill_live_event(
    name: str,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> str:
    parts = [f"[实时][Skill] {name}"]
    if payload.get("explicit_skill_name"):
        parts.append(f"explicit=/{payload['explicit_skill_name']}")
    if payload.get("skill_name"):
        parts.append(f"skill={payload['skill_name']}")
    if payload.get("name"):
        parts.append(f"skill={payload['name']}")
    if payload.get("relative_path"):
        parts.append(f"path={payload['relative_path']}")
    if payload.get("available_skill_count") is not None:
        parts.append(f"available={payload['available_skill_count']}")
    if payload.get("skill_context_chars") is not None:
        parts.append(f"context_chars={payload['skill_context_chars']}")
    if payload.get("result_preview"):
        parts.append(
            f'result="{_compact_live_value(payload["result_preview"], max_chars=120)}"'
        )
    return " ".join(parts) + _format_live_level(record) + _format_live_elapsed(
        record,
        payload,
    )


def _format_mcp_live_event(
    name: str,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> str:
    parts = [f"[实时][MCP] {name}"]
    if payload.get("server_names"):
        parts.append("servers=" + _compact_live_value(payload["server_names"]))
    if payload.get("tool_names"):
        parts.append("tools=" + _compact_live_value(payload["tool_names"]))
    if payload.get("tool_name"):
        parts.append(f"tool={payload['tool_name']}")
    if payload.get("args") is not None:
        parts.append("args=" + _compact_live_value(payload["args"], max_chars=160))
    if payload.get("result_preview"):
        parts.append(
            f'result="{_compact_live_value(payload["result_preview"], max_chars=120)}"'
        )
    return " ".join(parts) + _format_live_level(record) + _format_live_elapsed(
        record,
        payload,
    )


def _format_tool_live_event(
    name: str,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> str:
    category = str(payload.get("tool_category") or "")
    if category not in {"rag", "skill", "mcp"}:
        return ""
    label = {"rag": "RAG", "skill": "Skill", "mcp": "MCP"}[category]
    phase = "start" if name.endswith("_start") else "end"
    parts = [f"[实时][{label}] tool.{phase}", f"tool={payload.get('tool_name', '')}"]
    if phase == "start" and payload.get("args") is not None:
        parts.append("args=" + _compact_live_value(payload["args"], max_chars=160))
    if phase == "end" and payload.get("result_preview"):
        parts.append(
            f'result="{_compact_live_value(payload["result_preview"], max_chars=120)}"'
        )
    return " ".join(parts) + _format_live_level(record) + _format_live_elapsed(
        record,
        payload,
    )


def format_cli_live_event(record: dict[str, Any]) -> str:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    event_type = str(record.get("type") or "")
    name = str(record.get("name") or "")
    if event_type == "rag":
        return _format_rag_live_event(name, payload, record)
    if event_type in {"query_rewrite", "retrieval"}:
        return _format_rag_live_event(name, payload, record)
    if event_type == "memory":
        return _format_memory_live_event(name, payload, record)
    if event_type == "skill":
        return _format_skill_live_event(name, payload, record)
    if event_type == "mcp":
        return _format_mcp_live_event(name, payload, record)
    if event_type == "tool" and name in {
        "agent.tool_call_start",
        "agent.tool_call_end",
    }:
        return _format_tool_live_event(name, payload, record)
    if event_type == "tool" and name == "tool.search_product_knowledge":
        return _format_rag_live_event(name, payload, record)
    return ""


class CLILiveEventPrinter:
    """Print selected trace records immediately during an interactive CLI turn."""

    def __init__(self, enabled: bool = DEFAULT_LIVE_EVENTS_ENABLED) -> None:
        self.enabled = enabled

    def __call__(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = format_cli_live_event(record)
        if line:
            print(line, flush=True)


MEMORY_LAYER_LABELS = {
    "profile": "用户画像与稳定偏好",
    "episode": "历史事件摘要",
    "procedure": "可复用流程记忆",
}


def format_memory_context(memories: list[dict]) -> str:
    if not memories:
        return ""

    blocks: list[str] = []
    for index, item in enumerate(memories, start=1):
        blocks.append(
            "\n".join(
                [
                    f"记忆{index}",
                    f"类型: {item['memory_type']}",
                    f"重要性: {item['importance']:.2f}",
                    f"内容: {item['content']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_layered_memory_context(layered_memories: dict[str, list[dict]]) -> str:
    blocks: list[str] = []
    for layer in ("profile", "episode", "procedure"):
        memories = layered_memories.get(layer) or []
        if not memories:
            continue

        layer_blocks = []
        for index, item in enumerate(memories, start=1):
            layer_blocks.append(
                "\n".join(
                    [
                        f"记忆{index}",
                        f"类型: {item['memory_type']}",
                        f"重要性: {item['importance']:.2f}",
                        f"内容: {item['content']}",
                    ]
                )
            )
        blocks.append(
            "\n".join(
                [
                    f"## {MEMORY_LAYER_LABELS.get(layer, layer)}",
                    *layer_blocks,
                ]
            )
        )
    return "\n\n".join(blocks)


def format_memory_list(memories: list[dict]) -> str:
    if not memories:
        return "当前没有长期记忆。"

    lines = []
    for item in memories:
        lines.append(
            f"{item['id'][:8]}  [{item.get('memory_layer', 'profile')}/"
            f"{item['memory_type']}] "
            f"{item['content']}  importance={item['importance']:.2f}"
        )
    return "\n".join(lines)


def resolve_memory_id(
    memory_service: MemoryService,
    user_id: str,
    memory_ref: str,
) -> tuple[str | None, str | None]:
    memories = memory_service.list_memories(user_id, limit=200)
    matches = [
        item["id"]
        for item in memories
        if item["id"] == memory_ref or item["id"].startswith(memory_ref)
    ]
    if not matches:
        return None, "没有找到这条记忆。"
    if len(matches) > 1:
        return None, "这个 ID 前缀匹配到多条记忆，请多输入几位。"
    return matches[0], None


def handle_memory_command(
    user_input: str,
    memory_service: MemoryService | None,
    user_id: str,
) -> bool:
    command, _, argument = user_input.partition(" ")
    if command not in {
        "/memory",
        "/remember",
        "/remember-procedure",
        "/remember-episode",
        "/forget",
        "/clear-memory",
    }:
        return False

    if memory_service is None:
        print("\n记忆系统当前已关闭。")
        return True

    try:
        if command == "/memory":
            print(f"\n{format_memory_list(memory_service.list_memories(user_id))}")
            return True

        if command in {"/remember", "/remember-procedure", "/remember-episode"}:
            content = argument.strip()
            if not content:
                print(f"\n用法: {command} 需要记住的内容")
                return True
            memory_type = {
                "/remember": "instruction",
                "/remember-procedure": "procedure",
                "/remember-episode": "episode",
            }[command]
            record = memory_service.add_memory(
                user_id,
                content,
                memory_type=memory_type,
                importance=0.8,
                source="manual",
            )
            print(f"\n已记住: {record['id'][:8]} {record['content']}")
            return True

        if command == "/forget":
            memory_ref = argument.strip()
            if not memory_ref:
                print("\n用法: /forget 记忆ID前缀")
                return True
            memory_id, error = resolve_memory_id(memory_service, user_id, memory_ref)
            if error is not None or memory_id is None:
                print(f"\n{error}")
                return True
            memory_service.forget_memory(memory_id, user_id=user_id)
            print("\n已删除这条记忆。")
            return True

        if command == "/clear-memory":
            count = memory_service.clear_user_memory(user_id)
            print(f"\n已清空 {count} 条长期记忆。")
            return True

    except Exception as error:
        print(f"\n记忆操作失败：{error!r}")
        return True

    return True


def build_tool_map(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """Build a tool lookup and fail early on duplicate tool names."""
    tool_map: dict[str, BaseTool] = {}
    duplicates: list[str] = []
    for item in tools:
        if item.name in tool_map:
            duplicates.append(item.name)
        tool_map[item.name] = item

    if duplicates:
        duplicate_names = ", ".join(sorted(set(duplicates)))
        raise ValueError(
            "Duplicate tool names detected. "
            f"Enable MCP tool_name_prefix or rename tools: {duplicate_names}"
        )
    return tool_map


def tool_call_signature(tool_calls: list[dict[str, Any]]) -> str:
    comparable = [
        {
            "name": item.get("name"),
            "args": item.get("args") or {},
        }
        for item in tool_calls
    ]
    return json.dumps(comparable, ensure_ascii=False, sort_keys=True, default=str)


def next_repeated_tool_call_count(
    state: AgentState,
    tool_calls: list[dict[str, Any]],
) -> int:
    signature = tool_call_signature(tool_calls)
    if signature and signature == state.get("last_tool_call_signature", ""):
        return int(state.get("repeated_tool_call_count", 0)) + 1
    return 1


def build_agent(
    rag: RAGService,
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    agent_provider: str = DEFAULT_CHAT_PROVIDER,
    agent_model_name: str = DEFAULT_AGENT_MODEL,
    agent_model_kwargs: dict[str, Any] | None = None,
    rewrite_provider: str = DEFAULT_CHAT_PROVIDER,
    rewrite_model_name: str = DEFAULT_AGENT_MODEL,
    rewrite_model_kwargs: dict[str, Any] | None = None,
    memory_service: MemoryService | None = None,
    memory_extractor: LLMMemoryExtractor | None = None,
    memory_top_k: int = 5,
    skills_enabled: bool = True,
    skill_registry: SkillRegistry | None = None,
    mcp_tools: list[BaseTool] | None = None,
    trace_recorder: TraceRecorder | None = None,
    agent_model: Any | None = None,
    llm_retry_policy: LLMRetryPolicy | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    max_repeated_tool_calls: int = DEFAULT_MAX_REPEATED_TOOL_CALLS,
    reflection_enabled: bool = DEFAULT_REFLECTION_ENABLED,
):
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)
    retry_policy = llm_retry_policy or LLMRetryPolicy()
    max_tool_rounds = max(0, max_tool_rounds)
    max_repeated_tool_calls = max(1, max_repeated_tool_calls)
    rewriter = None
    if actual_query_rewrite_mode != "off":
        rewriter = LLMQueryRewriter(
            provider=rewrite_provider,
            model_name=rewrite_model_name,
            model_kwargs=rewrite_model_kwargs,
            trace_recorder=trace_recorder,
            retry_policy=retry_policy,
        )

    retrieval_tool = build_retrieval_tool_with_rewrite(
        rag,
        query_rewrite_mode=actual_query_rewrite_mode,
        rewriter=rewriter,
        trace_recorder=trace_recorder,
    )
    actual_skill_registry = skill_registry if skills_enabled else None
    skill_tools = (
        build_skill_tools(actual_skill_registry, trace_recorder=trace_recorder)
        if actual_skill_registry is not None
        else []
    )
    tools = [retrieval_tool, *skill_tools, *(mcp_tools or [])]
    tool_map = build_tool_map(tools)
    mcp_tool_names = {tool.name for tool in (mcp_tools or [])}

    def tool_category(tool_name: str) -> str:
        if tool_name == "search_product_knowledge":
            return "rag"
        if tool_name in {"load_skill", "read_skill_file"}:
            return "skill"
        if tool_name in mcp_tool_names:
            return "mcp"
        return "tool"

    base_model = agent_model or create_chat_model(
        provider=agent_provider,
        model_name=agent_model_name,
        **(agent_model_kwargs or {}),
    )
    model = base_model.bind_tools(tools)
    reflection_agent = (
        ReflectionAgent(
            rag=rag,
            provider=agent_provider,
            model_name=agent_model_name,
            model_kwargs=agent_model_kwargs,
            model=base_model,
            retry_policy=retry_policy,
            trace_recorder=trace_recorder,
        )
        if reflection_enabled
        else None
    )

    system_prompt = SystemMessage(
        content=(
            "你是一个通用电商客服 Agent。"
            "你可以服务任意商品类目，但不能臆测商品参数、库存、政策或售后规则。"
            "当用户问题涉及商品信息、尺码、材质、颜色、洗护、售后政策等事实内容时，"
            "优先调用知识库检索工具。"
            "如果知识库没有足够信息，就明确告知用户当前无法确认，"
            "不要编造答案，不要把 docs 文件名当成商品事实，不要做与客服无关的扩展。"
            "如果提供了用户长期记忆，只把它当成用户偏好或历史信息，不能当成商品事实。"
            "如果提供了 Skills 元数据或已加载 Skill 内容，按 Skill 指令处理对应任务；"
            "Skill 指令不能覆盖商品事实必须来自知识库这一原则。"
            "如果启用了 MCP 工具，它们只用于查询或操作外部系统；"
            "不要把 MCP 返回内容与商品知识库事实混淆。"
            "回答保持简洁、自然、客服口吻。"
        )
    )

    def load_memory(state: AgentState) -> AgentState:
        user_id = state.get("user_id", DEFAULT_USER_ID)
        if memory_service is None:
            return {
                "user_id": user_id,
                "latest_user_message": latest_human_text(state.get("messages", [])),
                "memory_context": "",
            }

        user_message = latest_human_text(state.get("messages", []))
        if not user_message:
            return {
                "user_id": user_id,
                "latest_user_message": "",
                "memory_context": "",
            }

        try:
            layered_memories = memory_service.search_memory_layers(
                user_id,
                user_message,
                layer_top_k={
                    "profile": memory_top_k,
                    "episode": max(1, memory_top_k // 2),
                    "procedure": max(1, memory_top_k // 2),
                },
            )
        except Exception:
            layered_memories = {}

        if trace_recorder is not None:
            trace_recorder.event(
                "memory",
                "agent.load_memory",
                {
                    "user_id": user_id,
                    "query": user_message,
                    "layer_counts": {
                        layer: len(items)
                        for layer, items in layered_memories.items()
                    },
                },
            )
        return {
            "user_id": user_id,
            "latest_user_message": user_message,
            "memory_context": format_layered_memory_context(layered_memories),
        }

    def load_skills(state: AgentState) -> AgentState:
        if actual_skill_registry is None:
            return {"skill_context": ""}

        discovery_prompt = actual_skill_registry.discovery_prompt()
        user_message = latest_human_text(state.get("messages", []))
        explicit_skill_name = actual_skill_registry.explicit_invocation_name(
            user_message
        )
        explicit_context = actual_skill_registry.render_explicit_skill_context(
            user_message
        )
        blocks = [item for item in [discovery_prompt, explicit_context] if item]
        update: AgentState = {"skill_context": "\n\n".join(blocks)}
        if explicit_skill_name is not None:
            update["active_skill_names"] = [explicit_skill_name]
        if trace_recorder is not None:
            available_skills = actual_skill_registry.list_skills()
            trace_recorder.event(
                "skill",
                "agent.load_skills",
                {
                    "available_skill_count": len(available_skills),
                    "available_skill_names": [
                        skill.name for skill in available_skills
                    ],
                    "explicit_skill_name": explicit_skill_name,
                    "skill_context_chars": len(update["skill_context"]),
                },
            )
        return update

    def trace_model_retry_failure(event: dict[str, Any]) -> None:
        if trace_recorder is None:
            return
        trace_recorder.event(
            "model",
            "agent.model_retry",
            event,
            level="warning" if event.get("will_retry") else "error",
        )

    async def call_model(state: AgentState) -> AgentState:
        prompt_messages: list[BaseMessage] = [system_prompt]
        memory_context = state.get("memory_context", "")
        if memory_context:
            prompt_messages.append(
                SystemMessage(
                    content=(
                        "以下是与当前用户问题可能相关的长期记忆。"
                        "这些内容已按 profile、episode、procedure 分层；"
                        "它们只代表用户偏好、约束、历史信息或可复用流程，不代表商品知识库事实；"
                        "只有与当前问题相关时才使用。\n\n"
                        f"{memory_context}"
                    )
                )
            )
        skill_context = state.get("skill_context", "")
        if skill_context:
            prompt_messages.append(SystemMessage(content=skill_context))

        if trace_recorder is not None:
            trace_recorder.event(
                "model",
                "agent.model_call_start",
                {
                    "prompt_message_count": len(prompt_messages),
                    "conversation_message_count": len(state["messages"]),
                },
            )
        model_messages = [*prompt_messages, *state["messages"]]
        response = await ainvoke_with_retry(
            lambda: model.ainvoke(model_messages),
            retry_policy=retry_policy,
            operation="agent.model_ainvoke",
            on_failure=trace_model_retry_failure,
        )
        if trace_recorder is not None:
            trace_recorder.event(
                "model",
                "agent.model_call_end",
                {
                    "has_tool_calls": bool(
                        isinstance(response, AIMessage) and response.tool_calls
                    ),
                    "tool_call_count": (
                        len(response.tool_calls)
                        if isinstance(response, AIMessage) and response.tool_calls
                        else 0
                    ),
                    "content_preview": preview_text(
                        coerce_message_content(response.content)
                    ),
                    **message_usage_metadata(response),
                },
            )
        if (
            reflection_agent is not None
            and isinstance(response, AIMessage)
            and not response.tool_calls
        ):
            initial_answer = coerce_message_content(response.content).strip()
            revised_answer = await reflection_agent.review_and_revise(
                user_question=state.get("latest_user_message", "")
                or latest_human_text(state.get("messages", [])),
                initial_answer=initial_answer,
                evidence_context=format_recent_tool_context(
                    state.get("messages", [])
                ),
                memory_context=state.get("memory_context", ""),
                skill_context=state.get("skill_context", ""),
            )
            if revised_answer != initial_answer:
                response = replace_ai_message_content(response, revised_answer)
        return {"messages": [response]}

    def route_tools(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            tool_round_count = int(state.get("tool_round_count", 0))
            repeated_tool_calls = next_repeated_tool_call_count(
                state,
                last_message.tool_calls,
            )
            if (
                tool_round_count >= max_tool_rounds
                or repeated_tool_calls > max_repeated_tool_calls
            ):
                return "loop_guard"
            return "tools"
        return "save_memory"

    def stop_tool_loop(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        tool_calls = (
            last_message.tool_calls
            if isinstance(last_message, AIMessage) and last_message.tool_calls
            else []
        )
        tool_round_count = int(state.get("tool_round_count", 0))
        repeated_tool_calls = next_repeated_tool_call_count(state, tool_calls)
        reason = (
            "max_tool_rounds"
            if tool_round_count >= max_tool_rounds
            else "repeated_tool_calls"
        )
        tool_names = [str(item.get("name", "")) for item in tool_calls]
        if trace_recorder is not None:
            trace_recorder.event(
                "agent",
                "agent.tool_loop_guard",
                {
                    "reason": reason,
                    "tool_round_count": tool_round_count,
                    "max_tool_rounds": max_tool_rounds,
                    "repeated_tool_call_count": repeated_tool_calls,
                    "max_repeated_tool_calls": max_repeated_tool_calls,
                    "tool_names": tool_names,
                },
                level="warning",
            )
        aborted_tool_messages = [
            ToolMessage(
                content="工具调用已中止：触发循环保护。",
                tool_call_id=str(item.get("id", "")),
            )
            for item in tool_calls
            if item.get("id")
        ]
        return {
            "messages": [
                *aborted_tool_messages,
                AIMessage(
                    content=(
                        "抱歉，我这边连续调用工具后仍无法稳定完成本次处理，"
                        "为避免无效循环已先停止。请稍后再试，或把问题拆得更具体一些。"
                    )
                ),
            ]
        }

    async def call_tools(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        tool_messages = []
        active_skill_names = list(state.get("active_skill_names", []))
        current_signature = tool_call_signature(last_message.tool_calls)
        repeated_tool_call_count = next_repeated_tool_call_count(
            state,
            last_message.tool_calls,
        )
        tool_round_count = int(state.get("tool_round_count", 0)) + 1

        def allowed_tool_names() -> set[str] | None:
            if actual_skill_registry is None or not active_skill_names:
                return None

            names: set[str] = set()
            for skill_name in active_skill_names:
                skill = actual_skill_registry.get_skill(skill_name)
                if skill is None or not skill.allowed_tools:
                    continue
                names.update(skill.allowed_tools)

            if not names:
                return None
            return names | {"load_skill", "read_skill_file"}

        for tool_call in last_message.tool_calls:
            selected_tool = tool_map.get(tool_call["name"])
            if selected_tool is None:
                tool_result = f"未知工具: {tool_call['name']}"
            elif (
                allowed_tool_names() is not None
                and selected_tool.name not in allowed_tool_names()
            ):
                allowed = ", ".join(sorted(allowed_tool_names() or []))
                tool_result = (
                    f"当前已激活 skill 限制可用工具为: {allowed}。"
                    f"已拒绝调用: {selected_tool.name}"
                )
            else:
                if trace_recorder is not None:
                    trace_recorder.event(
                        "tool",
                        "agent.tool_call_start",
                        {
                            "tool_name": selected_tool.name,
                            "tool_category": tool_category(selected_tool.name),
                            "args": tool_call.get("args"),
                        },
                    )
                tool_result = await selected_tool.ainvoke(tool_call["args"])
                if trace_recorder is not None:
                    trace_recorder.event(
                        "tool",
                        "agent.tool_call_end",
                        {
                            "tool_name": selected_tool.name,
                            "tool_category": tool_category(selected_tool.name),
                            "result_preview": preview_text(tool_result),
                        },
                    )
                if selected_tool.name == "load_skill" and isinstance(
                    tool_call.get("args"),
                    dict,
                ):
                    loaded_skill_name = str(
                        tool_call["args"].get("name") or ""
                    ).strip().lower()
                    if (
                        actual_skill_registry is not None
                        and actual_skill_registry.get_skill(loaded_skill_name)
                        is not None
                        and loaded_skill_name not in active_skill_names
                    ):
                        active_skill_names.append(loaded_skill_name)
            tool_messages.append(
                {
                    "role": "tool",
                    "content": (
                        tool_result
                        if isinstance(tool_result, str | list)
                        else str(tool_result)
                    ),
                    "tool_call_id": tool_call["id"],
                }
            )
        return {
            "messages": tool_messages,
            "active_skill_names": active_skill_names,
            "tool_round_count": tool_round_count,
            "last_tool_call_signature": current_signature,
            "repeated_tool_call_count": repeated_tool_call_count,
        }

    def save_memory(state: AgentState) -> AgentState:
        if memory_service is None or memory_extractor is None:
            return {}

        messages = state.get("messages", [])
        if not messages:
            return {}

        final_message = latest_ai_message(messages)
        if final_message is None or final_message.tool_calls:
            if trace_recorder is not None:
                trace_recorder.event(
                    "memory",
                    "agent.save_memory_skipped",
                    {"reason": "no_final_ai_message_or_pending_tool_calls"},
                )
            return {}

        user_id = state.get("user_id", DEFAULT_USER_ID)
        user_message = (
            state.get("latest_user_message", "").strip()
            or latest_human_text(messages)
        )
        assistant_message = coerce_message_content(final_message.content).strip()
        if not user_message or not assistant_message:
            if trace_recorder is not None:
                trace_recorder.event(
                    "memory",
                    "agent.save_memory_skipped",
                    {
                        "reason": "empty_user_or_assistant_message",
                        "has_user_message": bool(user_message),
                        "has_assistant_message": bool(assistant_message),
                    },
                )
            return {}

        try:
            try:
                existing_memories = memory_service.search_memory(
                    user_id,
                    user_message,
                    top_k=8,
                )
            except Exception as error:
                existing_memories = []
                if trace_recorder is not None:
                    trace_recorder.event(
                        "memory",
                        "agent.search_existing_memory_failed",
                        {"user_id": user_id, "error": repr(error)},
                        level="warning",
                    )
            extracted_memories = memory_extractor.extract(
                user_message=user_message,
                assistant_message=assistant_message,
                existing_memories=existing_memories,
            )

            if not extracted_memories:
                if trace_recorder is not None:
                    trace_recorder.event(
                        "memory",
                        "agent.save_memory_skipped",
                        {"reason": "extractor_returned_empty", "user_id": user_id},
                    )
                return {}

            known_contents = {
                item["content"]
                for item in memory_service.list_memories(user_id, limit=200)
            }
            new_memories = [
                {
                    "content": item.content,
                    "memory_type": item.memory_type,
                    "importance": item.importance,
                    "source": "conversation",
                    "expires_at": item.expires_at,
                }
                for item in extracted_memories
                if item.content not in known_contents
            ]
            if new_memories:
                memory_service.add_memories(user_id, new_memories)
                if trace_recorder is not None:
                    trace_recorder.event(
                        "memory",
                        "agent.save_memory",
                        {
                            "user_id": user_id,
                            "new_memory_count": len(new_memories),
                            "memory_types": [
                                item["memory_type"] for item in new_memories
                            ],
                        },
                    )
        except Exception as error:
            if trace_recorder is not None:
                trace_recorder.event(
                    "memory",
                    "agent.save_memory_failed",
                    {"user_id": user_id, "error": repr(error)},
                    level="error",
                )
            return {}

        return {}

    graph = StateGraph(AgentState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("load_skills", load_skills)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("loop_guard", stop_tool_loop)
    graph.add_node("save_memory", save_memory)
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "load_skills")
    graph.add_edge("load_skills", "agent")
    graph.add_conditional_edges(
        "agent",
        route_tools,
        {
            "tools": "tools",
            "loop_guard": "loop_guard",
            "save_memory": "save_memory",
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("loop_guard", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile(), base_model, system_prompt


async def run_cli_async(
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    agent_provider: str = DEFAULT_CHAT_PROVIDER,
    agent_model_name: str = DEFAULT_AGENT_MODEL,
    agent_model_kwargs: dict[str, Any] | None = None,
    rewrite_provider: str = DEFAULT_CHAT_PROVIDER,
    rewrite_model_name: str = DEFAULT_AGENT_MODEL,
    rewrite_model_kwargs: dict[str, Any] | None = None,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedding_model_kwargs: dict[str, Any] | None = None,
    reranker_provider: str = DEFAULT_RERANKER_PROVIDER,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    reranker_model_kwargs: dict[str, Any] | None = None,
    reranker_device: str | None = None,
    reranker_batch_size: int = 16,
    bm25_enabled: bool = True,
    cross_encoder_enabled: bool = False,
    user_id: str = DEFAULT_USER_ID,
    memory_enabled: bool = True,
    memory_provider: str = DEFAULT_CHAT_PROVIDER,
    memory_model_name: str = DEFAULT_AGENT_MODEL,
    memory_model_kwargs: dict[str, Any] | None = None,
    memory_top_k: int = 5,
    skills_enabled: bool = True,
    skill_dirs: list[str] | None = None,
    mcp_enabled: bool = False,
    mcp_config_path: str = DEFAULT_MCP_CONFIG_PATH,
    trace_enabled: bool = False,
    live_events_enabled: bool = DEFAULT_LIVE_EVENTS_ENABLED,
    show_config: bool = DEFAULT_CLI_CONFIG_OUTPUT_ENABLED,
    trace_dir: str = DEFAULT_TRACE_DIR,
    data_dir: str = "data",
    memory_dir: str = "memory",
    llm_retry_attempts: int = 3,
    llm_timeout_s: float | None = 30.0,
    llm_retry_backoff_s: float = 1.0,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    max_repeated_tool_calls: int = DEFAULT_MAX_REPEATED_TOOL_CALLS,
    reflection_enabled: bool = DEFAULT_REFLECTION_ENABLED,
) -> None:
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)
    llm_retry_policy = LLMRetryPolicy(
        max_attempts=llm_retry_attempts,
        per_attempt_timeout_s=llm_timeout_s,
        initial_backoff_s=llm_retry_backoff_s,
    )
    trace_recorder = None
    if trace_enabled or live_events_enabled:
        trace_recorder = TraceRecorder(
            trace_dir=trace_dir,
            enabled=trace_enabled,
            default_tags={
                "entrypoint": "cli",
                "user_id": user_id,
                "agent_provider": agent_provider,
                "agent_model": agent_model_name,
            },
            event_sinks=[
                CLILiveEventPrinter(enabled=live_events_enabled),
            ],
        )
    if trace_recorder is not None:
        trace_recorder.event(
            "runtime",
            "cli.startup",
            {
                "query_rewrite_mode": actual_query_rewrite_mode,
                "agent_provider": agent_provider,
                "agent_model_name": agent_model_name,
                "rewrite_provider": rewrite_provider,
                "rewrite_model_name": rewrite_model_name,
                "bm25_enabled": bm25_enabled,
                "cross_encoder_enabled": cross_encoder_enabled,
                "embedding_provider": embedding_provider,
                "embedding_model_name": embedding_model_name,
                "reranker_provider": reranker_provider,
                "reranker_model_name": reranker_model_name,
                "reranker_device": reranker_device,
                "reranker_batch_size": reranker_batch_size,
                "memory_enabled": memory_enabled,
                "memory_provider": memory_provider,
                "memory_model_name": memory_model_name,
                "memory_top_k": memory_top_k,
                "skills_enabled": skills_enabled,
                "skill_dirs": skill_dirs or [],
                "mcp_enabled": mcp_enabled,
                "mcp_config_path": mcp_config_path,
                "trace_dir": trace_dir,
                "trace_enabled": trace_enabled,
                "live_events_enabled": live_events_enabled,
                "show_config": show_config,
                "data_dir": data_dir,
                "memory_dir": memory_dir,
                "llm_retry_attempts": llm_retry_policy.normalized().max_attempts,
                "llm_timeout_s": llm_retry_policy.normalized().per_attempt_timeout_s,
                "llm_retry_backoff_s": (
                    llm_retry_policy.normalized().initial_backoff_s
                ),
                "max_tool_rounds": max_tool_rounds,
                "max_repeated_tool_calls": max_repeated_tool_calls,
                "reflection_enabled": reflection_enabled,
            },
        )
    rag = RAGService(
        data_dir=data_dir,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        embedding_model_kwargs=embedding_model_kwargs,
        reranker_provider=reranker_provider,
        reranker_model_name=reranker_model_name,
        reranker_model_kwargs=reranker_model_kwargs,
        reranker_device=reranker_device,
        reranker_batch_size=reranker_batch_size,
        default_use_bm25=bm25_enabled,
        default_use_rerank=cross_encoder_enabled,
        trace_recorder=trace_recorder,
    )
    memory_service = (
        MemoryService(
            data_dir=memory_dir,
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name,
            embedding_model_kwargs=embedding_model_kwargs,
            trace_recorder=trace_recorder,
        )
        if memory_enabled
        else None
    )
    memory_extractor = (
        LLMMemoryExtractor(
            provider=memory_provider,
            model_name=memory_model_name,
            model_kwargs=memory_model_kwargs,
            retry_policy=llm_retry_policy,
            trace_recorder=trace_recorder,
        )
        if memory_service is not None
        else None
    )
    skill_registry = (
        SkillRegistry.from_project_root(
            Path.cwd(),
            extra_skill_dirs=skill_dirs,
        )
        if skills_enabled
        else None
    )
    mcp_result = (
        await load_mcp_tools_from_config(mcp_config_path)
        if mcp_enabled
        else None
    )
    mcp_tools = mcp_result.tools if mcp_result is not None else []
    if trace_recorder is not None and mcp_result is not None:
        trace_recorder.event(
            "mcp",
            "mcp.load_tools",
            {
                "server_names": mcp_result.server_names,
                "tool_names": [tool.name for tool in mcp_tools],
                "tool_count": len(mcp_tools),
            },
        )
    app, model, system_prompt = build_agent(
        rag,
        query_rewrite_mode=actual_query_rewrite_mode,
        agent_provider=agent_provider,
        agent_model_name=agent_model_name,
        agent_model_kwargs=agent_model_kwargs,
        rewrite_provider=rewrite_provider,
        rewrite_model_name=rewrite_model_name,
        rewrite_model_kwargs=rewrite_model_kwargs,
        memory_service=memory_service,
        memory_extractor=memory_extractor,
        memory_top_k=memory_top_k,
        skills_enabled=skills_enabled,
        skill_registry=skill_registry,
        mcp_tools=mcp_tools,
        trace_recorder=trace_recorder,
        llm_retry_policy=llm_retry_policy,
        max_tool_rounds=max_tool_rounds,
        max_repeated_tool_calls=max_repeated_tool_calls,
        reflection_enabled=reflection_enabled,
    )
    messages: list[BaseMessage] = []

    print("电商客服 Agent 已启动，输入 quit 或 exit 结束。")
    if show_config:
        print(f"当前 Agent 模型: {agent_provider}:{agent_model_name}")
        print(f"当前 Embedding 模型: {embedding_provider}:{embedding_model_name}")
        print(f"当前 query 改写模式: {actual_query_rewrite_mode}")
        if actual_query_rewrite_mode != "off":
            print(f"当前 query 改写模型: {rewrite_provider}:{rewrite_model_name}")
        print(f"当前 BM25 模式: {'on' if bm25_enabled else 'off'}")
        print(f"当前 Rerank 模式: {'on' if cross_encoder_enabled else 'off'}")
        if cross_encoder_enabled:
            print(f"当前 Rerank 模型: {reranker_provider}:{reranker_model_name}")
        print(f"当前 data_dir: {data_dir}")
        print(f"当前 memory_dir: {memory_dir}")
        print(f"当前 memory 模式: {'on' if memory_enabled else 'off'}")
        if memory_enabled:
            print(f"当前 memory 抽取模型: {memory_provider}:{memory_model_name}")
        print(
            "当前 LLM 重试策略: "
            f"attempts={llm_retry_policy.normalized().max_attempts}, "
            f"timeout={llm_retry_policy.normalized().per_attempt_timeout_s}s, "
            f"backoff={llm_retry_policy.normalized().initial_backoff_s}s"
        )
        print(
            "当前工具循环保护: "
            f"max_tool_rounds={max_tool_rounds}, "
            f"max_repeated_tool_calls={max_repeated_tool_calls}"
        )
        print(f"当前 Reflection 模式: {'on' if reflection_enabled else 'off'}")
        if trace_enabled and trace_recorder is not None:
            print(f"当前 trace 模式: on ({trace_recorder.path})")
        else:
            print("当前 trace 模式: off")
        print(f"当前 CLI 实时事件: {'on' if live_events_enabled else 'off'}")
        if skill_registry is not None:
            skills = skill_registry.list_skills()
            print(f"当前 skills 模式: on ({len(skills)} 个)")
            if skills:
                print("可用 skills: " + ", ".join(skill.name for skill in skills))
            if skill_registry.errors:
                print("skills 加载警告:")
                for error in skill_registry.errors:
                    print(f"- {error}")
        else:
            print("当前 skills 模式: off")
        if mcp_result is not None:
            server_names = ", ".join(mcp_result.server_names) or "无"
            tool_names = ", ".join(tool.name for tool in mcp_tools) or "无"
            print(f"当前 MCP 模式: on ({len(mcp_tools)} 个工具)")
            print(f"MCP servers: {server_names}")
            print(f"MCP tools: {tool_names}")
        else:
            print("当前 MCP 模式: off")
        print(f"当前 user_id: {user_id}")
    try:
        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("客服会话已结束。")
                break
            if not user_input:
                continue
            if handle_memory_command(user_input, memory_service, user_id):
                continue

            input_messages = messages + [HumanMessage(content=user_input)]
            turn_start = time.perf_counter()
            if trace_recorder is not None:
                trace_recorder.event(
                    "agent",
                    "agent.user_turn_start",
                    {
                        "user_id": user_id,
                        "input_preview": preview_text(user_input),
                        "history_message_count": len(messages),
                    },
                )
            try:
                print("\n客服: ", end="", flush=True)
                streamed_content = ""
                final_message = None
                result = None
                async for event in app.astream(
                    {
                        "messages": input_messages,
                        "user_id": user_id,
                    },
                    stream_mode="updates",
                ):
                    for node_name, node_output in event.items():
                        if not isinstance(node_output, dict):
                            result = None
                            continue
                        node_messages = node_output.get("messages") or []
                        for msg in node_messages:
                            if not isinstance(msg, AIMessage):
                                continue
                            if msg.tool_calls:
                                continue
                            content = coerce_message_content(msg.content).strip()
                            if content and content != streamed_content:
                                new_text = content[len(streamed_content):]
                                print(new_text, end="", flush=True)
                                streamed_content = content
                                final_message = msg
                    result = node_output
                print(flush=True)
            except Exception as error:
                print(flush=True)
                if trace_recorder is not None:
                    trace_recorder.event(
                        "agent",
                        "agent.user_turn_error",
                        {
                            "user_id": user_id,
                            "error": repr(error),
                            "elapsed_ms": (
                                time.perf_counter() - turn_start
                            )
                            * 1000,
                        },
                        level="error",
                    )
                model_messages = [system_prompt, *input_messages]
                print(f"客服: {format_tongyi_error(model, model_messages, error)}")
                continue

            if result and "messages" in result:
                messages = result["messages"]
            else:
                messages = input_messages + (
                    [final_message] if final_message else []
                )

            if final_message is None:
                last = messages[-1] if messages else None
                if isinstance(last, AIMessage):
                    final_message = last

            if isinstance(final_message, AIMessage):
                if not streamed_content:
                    content = coerce_message_content(final_message.content)
                    print(content, flush=True)
                if trace_recorder is not None:
                    trace_recorder.event(
                        "agent",
                        "agent.user_turn_end",
                        {
                            "user_id": user_id,
                            "elapsed_ms": (
                                time.perf_counter() - turn_start
                            )
                            * 1000,
                            "message_count": len(messages),
                            "output_preview": preview_text(
                                final_message.content
                            ),
                            **message_usage_metadata(final_message),
                        },
                    )
    finally:
        if memory_service is not None:
            memory_service.close()


def run_cli(**kwargs: Any) -> None:
    asyncio.run(run_cli_async(**kwargs))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ecommerce customer-service CLI.")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional .toml or .json config file. "
            "RAG_SERVER_CONFIG is also supported."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="RAG index data directory. Overrides config/env when set.",
    )
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Long-term memory data directory. Overrides config/env when set.",
    )
    parser.add_argument(
        "--agent-provider",
        "--chat-provider",
        dest="agent_provider",
        default=None,
        help="Provider used by the agent chat model. Built-ins: tongyi, openai, or an import path.",
    )
    parser.add_argument(
        "--agent-model",
        "--chat-model",
        dest="agent_model",
        default=None,
        help="Model name used by the agent chat model. Overrides config/env when set.",
    )
    parser.add_argument(
        "--agent-model-kwargs",
        "--chat-model-kwargs",
        dest="agent_model_kwargs",
        default=None,
        help="JSON object passed to the agent chat model constructor.",
    )
    parser.add_argument(
        "--query-rewrite",
        choices=QUERY_REWRITE_MODES,
        default=None,
        help=(
            "Control retrieval query rewriting. "
            "'on' is an alias for 'multi_query'. Defaults to config value."
        ),
    )
    parser.add_argument(
        "--bm25",
        choices=["on", "off"],
        default=None,
        help="Enable or disable BM25 keyword retrieval. Defaults to config value.",
    )
    parser.add_argument(
        "--cross-encoder",
        choices=["on", "off"],
        default=None,
        help="Enable or disable CrossEncoder reranking. Defaults to config value.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Provider used by the embedding model. Built-ins: dashscope, openai, or an import path.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name. Defaults to config value.",
    )
    parser.add_argument(
        "--embedding-model-kwargs",
        default=None,
        help="JSON object passed to the embedding model constructor.",
    )
    parser.add_argument(
        "--reranker-provider",
        "--rerank-provider",
        dest="reranker_provider",
        default=None,
        help="Provider used by reranking. Built-ins: cross_encoder or an import path.",
    )
    parser.add_argument(
        "--reranker-model",
        "--rerank-model",
        dest="reranker_model",
        default=None,
        help="Reranker model name. Defaults to config value.",
    )
    parser.add_argument(
        "--reranker-model-kwargs",
        "--rerank-model-kwargs",
        dest="reranker_model_kwargs",
        default=None,
        help="JSON object passed to the reranker constructor.",
    )
    parser.add_argument(
        "--reranker-device",
        "--rerank-device",
        dest="reranker_device",
        default=None,
        help="Optional device for reranker loading, such as cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--reranker-batch-size",
        "--rerank-batch-size",
        dest="reranker_batch_size",
        type=int,
        default=None,
        help="Batch size used by reranker predict(). Defaults to config value.",
    )
    parser.add_argument(
        "--rewrite-provider",
        default=None,
        help="Provider used by the query rewrite model. Defaults to the agent provider.",
    )
    parser.add_argument(
        "--rewrite-model",
        default=None,
        help="Model name used by the query rewriter. Defaults to the agent model.",
    )
    parser.add_argument(
        "--rewrite-model-kwargs",
        default=None,
        help="JSON object passed to the query rewrite model constructor.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="User id used to scope long-term memories.",
    )
    parser.add_argument(
        "--memory",
        choices=["on", "off"],
        default=None,
        help="Enable or disable long-term memory. Defaults to config value.",
    )
    parser.add_argument(
        "--memory-provider",
        default=None,
        help="Provider used by the memory extractor model. Defaults to the agent provider.",
    )
    parser.add_argument(
        "--memory-model",
        default=None,
        help="Model name used by the memory extractor. Defaults to the agent model.",
    )
    parser.add_argument(
        "--memory-model-kwargs",
        default=None,
        help="JSON object passed to the memory extractor model constructor.",
    )
    parser.add_argument(
        "--memory-top-k",
        type=int,
        default=None,
        help="Number of long-term memories loaded per profile layer.",
    )
    parser.add_argument(
        "--skills",
        choices=["on", "off"],
        default=None,
        help="Enable or disable Anthropic-style skills. Defaults to config value.",
    )
    parser.add_argument(
        "--skills-dir",
        action="append",
        default=None,
        help=(
            "Additional Anthropic-style skills directory. "
            "Can be passed multiple times. Defaults to .claude/skills."
        ),
    )
    parser.add_argument(
        "--mcp",
        choices=["on", "off"],
        default=None,
        help="Enable or disable MCP client tools. Defaults to config value.",
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help=(
            "Path to MCP server JSON config. "
            f"Default config value is {DEFAULT_MCP_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--trace",
        choices=["on", "off"],
        default=None,
        help="Enable or disable JSONL runtime tracing. Defaults to config value.",
    )
    parser.add_argument(
        "--live-events",
        "--live-logs",
        dest="live_events",
        choices=["on", "off"],
        default=None,
        help=(
            "Show RAG, memory, skill, and MCP call logs live in the CLI. "
            "Defaults to config value."
        ),
    )
    parser.add_argument(
        "--show-config",
        choices=["on", "off"],
        default=None,
        help=(
            "Show the startup configuration summary in the CLI. "
            "Defaults to config value."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help=f"Directory for JSONL trace files. Defaults to {DEFAULT_TRACE_DIR}.",
    )
    parser.add_argument(
        "--llm-retry-attempts",
        type=int,
        default=None,
        help="Maximum attempts for each LLM call. Defaults to 3.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help=(
            "Per-attempt LLM timeout in seconds. "
            "Use 0 or a negative value to disable timeout. Defaults to 30."
        ),
    )
    parser.add_argument(
        "--llm-retry-backoff",
        type=float,
        default=None,
        help="Initial retry backoff in seconds. Defaults to 1.",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=None,
        help=(
            "Maximum Agent tool-call rounds per user turn. "
            f"Defaults to {DEFAULT_MAX_TOOL_ROUNDS}."
        ),
    )
    parser.add_argument(
        "--max-repeated-tool-calls",
        type=int,
        default=None,
        help=(
            "Maximum repeated identical tool-call rounds per user turn. "
            f"Defaults to {DEFAULT_MAX_REPEATED_TOOL_CALLS}."
        ),
    )
    parser.add_argument(
        "--reflection",
        choices=["on", "off"],
        default=None,
        help="Enable or disable post-answer reflection and correction.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _put_override(
    overrides: dict[str, dict[str, Any]],
    section: str,
    key: str,
    value: Any,
) -> None:
    if value is None:
        return
    overrides.setdefault(section, {})[key] = value


def build_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, dict[str, Any]] = {}
    _put_override(overrides, "paths", "data_dir", args.data_dir)
    _put_override(overrides, "paths", "memory_dir", args.memory_dir)
    _put_override(overrides, "paths", "trace_dir", args.trace_dir)
    _put_override(overrides, "paths", "mcp_config_path", args.mcp_config)
    _put_override(overrides, "agent", "provider", args.agent_provider)
    _put_override(overrides, "agent", "model", args.agent_model)
    _put_override(overrides, "agent", "model_kwargs", args.agent_model_kwargs)
    _put_override(overrides, "agent", "user_id", args.user_id)
    _put_override(overrides, "agent", "max_tool_rounds", args.max_tool_rounds)
    _put_override(
        overrides,
        "agent",
        "max_repeated_tool_calls",
        args.max_repeated_tool_calls,
    )
    if args.reflection is not None:
        _put_override(
            overrides,
            "agent",
            "reflection_enabled",
            args.reflection == "on",
        )
    _put_override(overrides, "retrieval", "query_rewrite", args.query_rewrite)
    if args.bm25 is not None:
        _put_override(overrides, "retrieval", "bm25", args.bm25 == "on")
    if args.cross_encoder is not None:
        _put_override(
            overrides,
            "retrieval",
            "cross_encoder",
            args.cross_encoder == "on",
        )
    _put_override(
        overrides,
        "retrieval",
        "embedding_provider",
        args.embedding_provider,
    )
    _put_override(overrides, "retrieval", "embedding_model", args.embedding_model)
    _put_override(
        overrides,
        "retrieval",
        "embedding_kwargs",
        args.embedding_model_kwargs,
    )
    _put_override(overrides, "retrieval", "reranker_provider", args.reranker_provider)
    _put_override(overrides, "retrieval", "reranker_model", args.reranker_model)
    _put_override(
        overrides,
        "retrieval",
        "reranker_kwargs",
        args.reranker_model_kwargs,
    )
    _put_override(overrides, "retrieval", "reranker_device", args.reranker_device)
    _put_override(
        overrides,
        "retrieval",
        "reranker_batch_size",
        args.reranker_batch_size,
    )
    _put_override(overrides, "llm", "rewrite_provider", args.rewrite_provider)
    _put_override(overrides, "llm", "rewrite_model", args.rewrite_model)
    _put_override(overrides, "llm", "rewrite_kwargs", args.rewrite_model_kwargs)
    _put_override(overrides, "llm", "memory_provider", args.memory_provider)
    _put_override(overrides, "llm", "memory_model", args.memory_model)
    _put_override(overrides, "llm", "memory_kwargs", args.memory_model_kwargs)
    _put_override(overrides, "llm", "retry_attempts", args.llm_retry_attempts)
    if args.llm_timeout is not None:
        overrides.setdefault("llm", {})["timeout_s"] = (
            args.llm_timeout if args.llm_timeout > 0 else None
        )
    _put_override(overrides, "llm", "retry_backoff_s", args.llm_retry_backoff)
    if args.memory is not None:
        _put_override(overrides, "memory", "enabled", args.memory == "on")
    _put_override(overrides, "memory", "top_k", args.memory_top_k)
    if args.skills is not None:
        _put_override(overrides, "skills", "enabled", args.skills == "on")
    _put_override(overrides, "skills", "dirs", args.skills_dir)
    if args.mcp is not None:
        _put_override(overrides, "mcp", "enabled", args.mcp == "on")
    if args.trace is not None:
        _put_override(overrides, "trace", "enabled", args.trace == "on")
    if args.live_events is not None:
        _put_override(overrides, "trace", "live", args.live_events == "on")
    if args.show_config is not None:
        _put_override(overrides, "cli", "show_config", args.show_config == "on")
    return overrides


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config = load_app_config(
            args.config,
            overrides=build_cli_overrides(args),
        )
    except ConfigError as error:
        raise SystemExit(f"配置错误: {error}") from error

    run_cli(**config.to_runtime_kwargs())


if __name__ == "__main__":
    main()
