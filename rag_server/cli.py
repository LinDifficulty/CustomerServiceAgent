from __future__ import annotations

import argparse
import asyncio
import difflib
import inspect
import os
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from pydantic import ValidationError

try:
    import readline as _readline
except ImportError:  # pragma: no cover - readline is unavailable on Windows.
    _readline = None

try:
    from prompt_toolkit import PromptSession as _PromptToolkitSession
    from prompt_toolkit.completion import Completer as _PTCompleter
    from prompt_toolkit.completion import Completion as _PTCompletion
    from prompt_toolkit.formatted_text import ANSI as _PTANSI
except ImportError:  # pragma: no cover - optional interactive enhancement.
    _PromptToolkitSession = None
    _PTCompleter = object
    _PTCompletion = None
    _PTANSI = None

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .cache_service import JsonCache, create_redis_cache
from .config import (
    DEFAULT_CLI_CONFIG_OUTPUT_ENABLED,
    DEFAULT_LIVE_EVENTS_ENABLED,
    DEFAULT_STREAM_OUTPUT_ENABLED,
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
from .query_rewrite import (
    LLMQueryRewriter,
    asearch_with_query_rewrites,
)
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
CLI_EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", "退出", "/退出"}
CLI_CLEAR_COMMANDS = {"clear", "/clear", "清空", "/清空"}
CLI_DISPLAY_NAME = "Tulip Agent"
CLI_PACKAGE_NAME = "rag-server"
CLI_VERSION_FALLBACK = "0.1.0"
CLI_LOGO = [
    ("   ██████   ", "purple"),
    ("███  ██  ███", "purple"),
    ("████████████", "purple"),
    ("   ██████   ", "purple"),
    ("     ██     ", "green"),
    ("███  ██  ███", "green"),
    ("  ████████  ", "green"),
    ("     ██     ", "green"),
]
MEMORY_SLASH_COMMANDS = {
    "/memory",
    "/remember",
    "/remember-procedure",
    "/remember-episode",
    "/forget",
    "/clear-memory",
}


@dataclass(frozen=True)
class SlashCommandSpec:
    command: str
    usage: str
    description: str
    category: str
    source: str = "builtin"

    @property
    def completion(self) -> str:
        return f"{self.command} "


BUILTIN_SLASH_COMMANDS = (
    SlashCommandSpec("/exit", "/exit", "结束会话", "Session"),
    SlashCommandSpec("/quit", "/quit", "结束会话", "Session"),
    SlashCommandSpec("/clear", "/clear", "清空当前会话上下文", "Session"),
    SlashCommandSpec("/help", "/help", "显示快捷帮助", "Session"),
    SlashCommandSpec("/memory", "/memory", "查看长期记忆", "Memory"),
    SlashCommandSpec("/remember", "/remember <内容>", "记录长期偏好或指令", "Memory"),
    SlashCommandSpec(
        "/remember-procedure",
        "/remember-procedure <内容>",
        "记录可复用流程",
        "Memory",
    ),
    SlashCommandSpec(
        "/remember-episode",
        "/remember-episode <内容>",
        "记录历史事件",
        "Memory",
    ),
    SlashCommandSpec("/forget", "/forget <ID前缀>", "删除一条长期记忆", "Memory"),
    SlashCommandSpec("/clear-memory", "/clear-memory", "清空当前用户的长期记忆", "Memory"),
)
SLASH_COMMAND_TOKEN_PATTERN = re.compile(r"^/[A-Za-z0-9][A-Za-z0-9-]*$")


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


def is_cli_exit_command(value: str) -> bool:
    return value.strip().casefold() in CLI_EXIT_COMMANDS


def is_cli_clear_command(value: str) -> bool:
    return value.strip().casefold() in CLI_CLEAR_COMMANDS


def cli_version() -> str:
    try:
        return version(CLI_PACKAGE_NAME)
    except PackageNotFoundError:
        return CLI_VERSION_FALLBACK


def should_use_cli_color(stream: Any | None = None) -> bool:
    if os.getenv("NO_COLOR") is not None:
        return False
    force_color = os.getenv("CLICOLOR_FORCE")
    if force_color and force_color != "0":
        return True
    target = stream if stream is not None else sys.stdout
    return bool(getattr(target, "isatty", lambda: False)())


def clear_terminal_startup(stream: Any | None = None) -> None:
    target = stream if stream is not None else sys.stdout
    if not getattr(target, "isatty", lambda: False)():
        return
    os.system("clear")


def terminal_width(default: int = 88) -> int:
    return max(52, min(shutil.get_terminal_size((default, 24)).columns, 140))


def format_cli_path(path: Path | str) -> str:
    absolute_path = Path(path).expanduser()
    try:
        absolute_path = absolute_path.resolve()
    except OSError:
        pass

    home = Path.home()
    try:
        relative_to_home = absolute_path.relative_to(home)
    except ValueError:
        return str(absolute_path)
    return f"~/{relative_to_home}"


class CLIStyle:
    """ANSI styling kept optional so captured test output stays plain."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    PROMPT = "\033[1;37m"
    MUTED = "\033[38;5;244m"
    ACCENT = "\033[38;5;173m"
    LOGO_PURPLE = "\033[38;5;177m"
    LOGO_GREEN = "\033[38;5;113m"
    WARNING = "\033[38;5;178m"
    ERROR = "\033[38;5;203m"

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = should_use_cli_color() if enabled is None else enabled

    def apply(self, text: str, *styles: str) -> str:
        if not self.enabled or not text:
            return text
        return f"{''.join(styles)}{text}{self.RESET}"

    def bold(self, text: str) -> str:
        return self.apply(text, self.BOLD)

    def dim(self, text: str) -> str:
        return self.apply(text, self.MUTED)

    def logo(self, text: str, color: str) -> str:
        styles = {
            "purple": self.LOGO_PURPLE,
            "green": self.LOGO_GREEN,
        }
        return self.apply(text, styles.get(color, self.ACCENT))

    def warning(self, text: str) -> str:
        return self.apply(text, self.WARNING)

    def error(self, text: str) -> str:
        return self.apply(text, self.ERROR)

    def prompt(self, text: str) -> str:
        return self.apply(text, self.PROMPT)


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


async def _rag_search_async(rag: Any, **kwargs: Any) -> list[dict]:
    if hasattr(rag, "asearch"):
        return await rag.asearch(**kwargs)
    return await asyncio.to_thread(rag.search, **kwargs)


async def _rewrite_async(
    rewriter: LLMQueryRewriter,
    question: str,
):
    if hasattr(rewriter, "arewrite"):
        return await rewriter.arewrite(question)
    return await asyncio.to_thread(rewriter.rewrite, question)


def build_retrieval_tool_with_rewrite(
    rag: RAGService,
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    rewriter: LLMQueryRewriter | None = None,
    trace_recorder: TraceRecorder | None = None,
):
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)

    @tool(description="检索商品知识，返回与用户问题最相关的商品信息片段。")
    async def search_product_knowledge(question: str) -> str:
        """检索商品知识库，返回与用户问题最相关的商品信息片段。"""
        trace_payload: dict[str, Any] = {
            "question": question,
            "query_rewrite_mode": actual_query_rewrite_mode,
        }
        if actual_query_rewrite_mode == "rewrite_only" and rewriter is not None:
            try:
                rewrite_result = await _rewrite_async(rewriter, question)
            except Exception as error:
                trace_payload["query_rewrite_error"] = repr(error)
                trace_payload["retrieval_queries"] = [question]
                results = await _rag_search_async(
                    rag,
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
                results = await _rag_search_async(
                    rag,
                    query=rewrite_result.rewritten_query,
                    top_k=3,
                    candidate_top_k=10,
                )
        elif actual_query_rewrite_mode == "multi_query" and rewriter is not None:
            try:
                rewrite_result = await _rewrite_async(rewriter, question)
            except Exception as error:
                trace_payload["query_rewrite_error"] = repr(error)
                trace_payload["retrieval_queries"] = [question]
                results = await _rag_search_async(
                    rag,
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
                results = await asearch_with_query_rewrites(
                    rag,
                    original_query=question,
                    rewritten_queries=deduplicated_queries,
                    top_k=3,
                    candidate_top_k=10,
                    trace_recorder=trace_recorder,
                )
        else:
            trace_payload["retrieval_queries"] = [question]
            results = await _rag_search_async(
                rag,
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


def coerce_graph_messages(messages: Any) -> list[BaseMessage]:
    if not isinstance(messages, list):
        return []
    return convert_to_messages(messages)


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


def add_ai_message_chunks(chunks: list[AIMessageChunk]) -> AIMessageChunk | None:
    merged: AIMessageChunk | None = None
    for chunk in chunks:
        if merged is None:
            merged = chunk
        else:
            merged = merged + chunk
    return merged


def ai_message_from_chunks(chunks: list[AIMessageChunk]) -> AIMessage:
    merged = add_ai_message_chunks(chunks)
    if merged is None:
        return AIMessage(content="")
    message = message_chunk_to_message(merged)
    if isinstance(message, AIMessage):
        return message
    return AIMessage(content=coerce_message_content(message.content))


def model_with_streaming_enabled(model: Any) -> Any:
    if getattr(model, "streaming", None) is True:
        return model
    for method_name in ("model_copy", "copy"):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        try:
            return method(update={"streaming": True})
        except Exception:
            continue
    try:
        setattr(model, "streaming", True)
    except Exception:
        return model
    return model


async def maybe_emit_output_delta(
    sink: Callable[[str], Any] | None,
    text: str,
) -> None:
    if sink is None or not text:
        return
    result = sink(text)
    if inspect.isawaitable(result):
        await result


class OutputDeltaDispatcher:
    def __init__(self) -> None:
        self.sink: Callable[[str], Any] | None = None

    async def __call__(self, text: str) -> None:
        await maybe_emit_output_delta(self.sink, text)


def format_tool_validation_error(
    tool_name: str,
    error: ValidationError,
    args: Any,
) -> str:
    if tool_name == "read_skill_file":
        return (
            "工具参数错误：read_skill_file 需要同时提供 name 和 relative_path。"
            "如果只是要读取 skill 的完整 SKILL.md，请改用 load_skill(name)。"
            "只有在 load_skill 返回的 Supporting files 列表中看到额外文件时，"
            "才调用 read_skill_file(name, relative_path)。"
            f"本次收到的参数: {args!r}。原始错误: {error}"
        )
    return f"工具参数错误：{tool_name} 收到的参数不符合 schema: {args!r}。原始错误: {error}"


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


def _format_live_status(record: dict[str, Any], payload: dict[str, Any]) -> str:
    parts: list[str] = []
    level = str(record.get("level") or "info")
    if level != "info":
        parts.append(level)
    elapsed = _live_elapsed_ms(record, payload)
    if elapsed is not None:
        parts.append(f"{elapsed:.1f}ms")
    return "  " + "  ".join(parts) if parts else ""


def _format_live_block(
    title: str,
    fields: list[tuple[str, Any, int]],
    record: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    lines = [f"  - {title}{_format_live_status(record, payload)}"]
    for label, value, max_chars in fields:
        if value is None or value == "":
            continue
        lines.append(f"    {label}: {_compact_live_value(value, max_chars=max_chars)}")
    return "\n".join(lines)


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
    return _format_live_block(
        f"RAG {name.removeprefix('tool.')}",
        [
            ("query", query, 90),
            ("rewrite", payload.get("rewritten_query"), 90),
            ("queries", payload.get("retrieval_queries"), 140),
            ("candidates", payload.get("candidate_count"), 40),
            ("results", payload.get("result_count"), 40),
            ("rerank", payload.get("use_rerank"), 40),
        ],
        record,
        payload,
    )


def _format_memory_live_event(
    name: str,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> str:
    layer_counts = payload.get("layer_counts")
    layers = (
        ", ".join(f"{layer}:{count}" for layer, count in sorted(layer_counts.items()))
        if isinstance(layer_counts, dict)
        else None
    )
    return _format_live_block(
        f"Memory {name}",
        [
            ("user", payload.get("user_id"), 80),
            ("query", payload.get("query"), 90),
            ("layers", layers, 120),
            ("types", payload.get("memory_types"), 120),
            ("results", payload.get("result_count"), 40),
            ("new", payload.get("new_memory_count"), 40),
            ("reason", payload.get("reason"), 120),
        ],
        record,
        payload,
    )


def _format_skill_live_event(
    name: str,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> str:
    explicit_skill = (
        f"/{payload['explicit_skill_name']}"
        if payload.get("explicit_skill_name")
        else None
    )
    return _format_live_block(
        f"Skill {name}",
        [
            ("explicit", explicit_skill, 80),
            ("skill", payload.get("skill_name") or payload.get("name"), 80),
            ("path", payload.get("relative_path"), 120),
            ("available", payload.get("available_skill_count"), 40),
            ("context chars", payload.get("skill_context_chars"), 40),
            ("result", payload.get("result_preview"), 120),
        ],
        record,
        payload,
    )


def _format_mcp_live_event(
    name: str,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> str:
    return _format_live_block(
        f"MCP {name}",
        [
            ("servers", payload.get("server_names"), 120),
            ("tools", payload.get("tool_names"), 140),
            ("tool", payload.get("tool_name"), 80),
            ("args", payload.get("args"), 160),
            ("result", payload.get("result_preview"), 120),
        ],
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
    return _format_live_block(
        f"{label} {payload.get('tool_name', '')} {phase}",
        [
            ("args", payload.get("args") if phase == "start" else None, 160),
            ("result", payload.get("result_preview") if phase == "end" else None, 120),
        ],
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

    def __init__(
        self,
        enabled: bool = DEFAULT_LIVE_EVENTS_ENABLED,
        view: "CLIView | None" = None,
    ) -> None:
        self.enabled = enabled
        self.view = view

    def __call__(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = format_cli_live_event(record)
        if line:
            if self.view is not None:
                self.view.print_live_event(line)
            else:
                print(line, flush=True)


class CLIStatusEventSink:
    """Drive the compact thinking status from actual runtime events."""

    def __init__(self, view: "CLIView") -> None:
        self.view = view

    def __call__(self, record: dict[str, Any]) -> None:
        indicator = self.view._thinking_indicator
        if indicator is None or not indicator.active:
            return
        status = self.status_for_record(record)
        if status:
            indicator.update(status)

    @staticmethod
    def status_for_record(record: dict[str, Any]) -> str:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        event_type = str(record.get("type") or "")
        name = str(record.get("name") or "")

        if (
            event_type == "tool"
            and name == "agent.tool_call_start"
            and payload.get("tool_category") == "rag"
        ):
            return "正在检索相关知识..."

        if (
            event_type == "tool"
            and name == "tool.search_product_knowledge"
        ):
            result_count = payload.get("result_count")
            if isinstance(result_count, int | float) and result_count > 0:
                return "找到相关知识..."
            if result_count == 0:
                return "未找到相关知识，继续直接回答..."

        if event_type in {"rag", "retrieval"}:
            result_count = payload.get("result_count")
            candidate_count = payload.get("candidate_count")
            if isinstance(result_count, int | float) and result_count > 0:
                return "找到相关知识..."
            if (
                result_count == 0
                and (candidate_count is None or candidate_count == 0)
            ):
                return "未找到相关知识，继续直接回答..."

        return ""


def builtin_slash_command_specs() -> list[SlashCommandSpec]:
    return list(BUILTIN_SLASH_COMMANDS)


def skill_slash_command_specs(
    skill_registry: SkillRegistry | None,
) -> list[SlashCommandSpec]:
    if skill_registry is None:
        return []
    return [
        SlashCommandSpec(
            command=f"/{skill.name}",
            usage=f"/{skill.name} <需求>",
            description=f"调用 skill：{skill.description}",
            category="Skills",
            source="skill",
        )
        for skill in skill_registry.list_skills()
        if skill.user_invocable
    ]


def available_slash_command_specs(
    skill_registry: SkillRegistry | None = None,
) -> list[SlashCommandSpec]:
    specs = [*builtin_slash_command_specs(), *skill_slash_command_specs(skill_registry)]
    return sorted(specs, key=lambda item: (item.category, item.command))


def is_potential_slash_command(value: str) -> bool:
    token = value.strip().split(maxsplit=1)[0] if value.strip() else ""
    return bool(token.startswith("/") and SLASH_COMMAND_TOKEN_PATTERN.fullmatch(token))


def find_slash_command_spec(
    command: str,
    *,
    skill_registry: SkillRegistry | None = None,
) -> SlashCommandSpec | None:
    normalized = command.strip().split(maxsplit=1)[0].lower()
    for item in available_slash_command_specs(skill_registry):
        if item.command == normalized:
            return item
    return None


def suggest_slash_commands(
    text: str,
    *,
    skill_registry: SkillRegistry | None = None,
    limit: int = 6,
) -> list[SlashCommandSpec]:
    stripped = text.strip()
    prefix = stripped.split(maxsplit=1)[0].lower() if stripped else "/"
    specs = available_slash_command_specs(skill_registry)
    if prefix == "/":
        return specs[:limit]

    startswith_matches = [
        item for item in specs if item.command.startswith(prefix)
    ]
    if startswith_matches:
        return startswith_matches[:limit]

    close_names = difflib.get_close_matches(
        prefix,
        [item.command for item in specs],
        n=limit,
        cutoff=0.35,
    )
    spec_by_name = {item.command: item for item in specs}
    return [spec_by_name[name] for name in close_names if name in spec_by_name]


def format_slash_command_table(specs: list[SlashCommandSpec]) -> str:
    if not specs:
        return ""

    lines: list[str] = []
    current_category = ""
    usage_width = min(max(len(item.usage) for item in specs), 30)
    for item in specs:
        if item.category != current_category:
            if lines:
                lines.append("")
            lines.append(f"{item.category}:")
            current_category = item.category
        lines.append(f"  {item.usage.ljust(usage_width)}  {item.description}")
    return "\n".join(lines)


def format_slash_command_suggestions(specs: list[SlashCommandSpec]) -> str:
    if not specs:
        return "没有匹配的 slash 命令。输入 /help 查看可用命令。"
    usage_width = min(max(len(item.usage) for item in specs), 30)
    return "\n".join(
        f"{item.usage.ljust(usage_width)}  {item.description}"
        for item in specs
    )


class CLICompleter:
    """Readline completer for slash commands and explicit skill invocation."""

    def __init__(self, specs: list[SlashCommandSpec]) -> None:
        self.specs = specs
        self._matches: list[str] = []

    def complete(self, text: str, state: int) -> str | None:
        if state == 0:
            self._matches = self._build_matches(text)
        try:
            return self._matches[state]
        except IndexError:
            return None

    def _build_matches(self, text: str) -> list[str]:
        if _readline is not None:
            buffer = _readline.get_line_buffer()
            begin = _readline.get_begidx()
        else:
            buffer = text
            begin = 0
        if begin != 0:
            return []
        prefix = (buffer or text).lower()
        if not prefix.startswith("/"):
            return []
        return [
            item.completion
            for item in self.specs
            if item.command.startswith(prefix)
        ]


def prompt_toolkit_slash_matches(
    specs: list[SlashCommandSpec],
    text_before_cursor: str,
) -> list[SlashCommandSpec]:
    if not text_before_cursor.startswith("/"):
        return []
    if any(character.isspace() for character in text_before_cursor):
        return []
    prefix = text_before_cursor.lower()
    return [item for item in specs if item.command.startswith(prefix)]


class PromptToolkitSlashCompleter(_PTCompleter):
    """Completion menu for prompt_toolkit's live slash command popup."""

    def __init__(self, specs: list[SlashCommandSpec]) -> None:
        self.specs = specs

    def get_completions(self, document: Any, complete_event: Any):
        if _PTCompletion is None:
            return
        text_before_cursor = str(getattr(document, "text_before_cursor", ""))
        matches = prompt_toolkit_slash_matches(self.specs, text_before_cursor)
        if not matches:
            return

        start_position = -len(text_before_cursor)
        for item in matches:
            yield _PTCompletion(
                item.completion,
                start_position=start_position,
                display=item.command,
                display_meta=item.description,
            )


class CLIInputSession:
    """Interactive input wrapper that enables optional readline completion."""

    def __init__(
        self,
        *,
        view: "CLIView",
        slash_commands: list[SlashCommandSpec],
        readline_module: Any | None = _readline,
        prompt_toolkit_session_factory: Any | None = _PromptToolkitSession,
        prompt_toolkit_session: Any | None = None,
        stdin: Any | None = None,
        stdout: Any | None = None,
    ) -> None:
        self.view = view
        self.slash_commands = slash_commands
        self.readline = readline_module
        self.prompt_toolkit_session_factory = prompt_toolkit_session_factory
        self._prompt_toolkit_session: Any | None = prompt_toolkit_session
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self._previous_completer: Any | None = None
        self._previous_delims: str | None = None
        self._configured = False

    def __enter__(self) -> CLIInputSession:
        self.enable_completion()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore_completion()

    def prompt(self) -> str:
        return input(self.view.input_prompt()).strip()

    async def prompt_async(self) -> str:
        if self.should_use_live_completion():
            message = self.view.input_prompt()
            if _PTANSI is not None:
                message = _PTANSI(message)
            return (
                await self.prompt_toolkit_session.prompt_async(
                    message,
                    completer=PromptToolkitSlashCompleter(self.slash_commands),
                    complete_while_typing=True,
                    reserve_space_for_menu=min(max(len(self.slash_commands), 3), 8),
                )
            ).strip()
        return await asyncio.to_thread(self.prompt)

    @property
    def prompt_toolkit_session(self) -> Any:
        if self._prompt_toolkit_session is None:
            self._prompt_toolkit_session = self.prompt_toolkit_session_factory()
        return self._prompt_toolkit_session

    def should_use_live_completion(self) -> bool:
        if self.prompt_toolkit_session_factory is None:
            return False
        stdin_isatty = getattr(self.stdin, "isatty", lambda: False)
        stdout_isatty = getattr(self.stdout, "isatty", lambda: False)
        return bool(stdin_isatty() and stdout_isatty())

    def enable_completion(self) -> bool:
        if self.should_use_live_completion():
            return True
        if self.readline is None:
            return False
        try:
            self._previous_completer = self.readline.get_completer()
            try:
                self._previous_delims = self.readline.get_completer_delims()
            except (AttributeError, OSError):
                self._previous_delims = None

            completer = CLICompleter(self.slash_commands)
            self.readline.set_completer(completer.complete)
            if hasattr(self.readline, "set_completer_delims"):
                delims = self._previous_delims or " \t\n"
                self.readline.set_completer_delims(delims.replace("/", ""))
            self._bind_tab_completion()
        except (AttributeError, OSError):
            return False
        self._configured = True
        return True

    def _bind_tab_completion(self) -> None:
        doc = str(getattr(self.readline, "__doc__", "") or "").lower()
        binding = "bind ^I rl_complete" if "libedit" in doc else "tab: complete"
        try:
            self.readline.parse_and_bind(binding)
        except (AttributeError, OSError):
            return

    def restore_completion(self) -> None:
        if not self._configured or self.readline is None:
            return
        try:
            self.readline.set_completer(self._previous_completer)
            if (
                self._previous_delims is not None
                and hasattr(self.readline, "set_completer_delims")
            ):
                self.readline.set_completer_delims(self._previous_delims)
        except (AttributeError, OSError):
            return


class CLIView:
    """Small presentation layer for the interactive CLI."""

    def __init__(
        self,
        *,
        style: CLIStyle | None = None,
        width: int | None = None,
    ) -> None:
        self.style = style or CLIStyle()
        self.width = width or terminal_width()
        self._thinking_indicator: CLIThinkingIndicator | None = None

    def _print(self, text: str = "", *, end: str = "\n", flush: bool = False) -> None:
        print(text, end=end, flush=flush)

    @staticmethod
    def _on_off(value: bool) -> str:
        return "on" if value else "off"

    @staticmethod
    def _render_pairs(pairs: list[tuple[str, Any]]) -> list[str]:
        visible = [(label, value) for label, value in pairs if value is not None]
        if not visible:
            return []
        width = max(len(label) for label, _ in visible)
        return [f"  {label.ljust(width)}  {value}" for label, value in visible]

    def _divider(self) -> str:
        return self.style.dim("-" * self.width)

    @staticmethod
    def _visible_center(text: str, width: int) -> str:
        """Center *text* within *width* columns, ignoring ANSI escape sequences."""
        visible = re.sub(r"\033\[[0-9;]*m", "", text)
        pad = max(0, (width - len(visible)) // 2)
        return " " * pad + text

    def _status(self, label: str, enabled: bool, extra: str | None = None) -> str:
        state = "on" if enabled else "off"
        text = f"{label}:{state}"
        if extra:
            text += f" ({extra})"
        return text

    def _short_tool_summary(
        self,
        *,
        skill_registry: SkillRegistry | None,
        mcp_result: Any | None,
        mcp_tools: list[BaseTool],
    ) -> str:
        skill_count = len(skill_registry.list_skills()) if skill_registry else 0
        mcp_count = len(mcp_tools) if mcp_result is not None else 0
        return f"skills:{skill_count} · mcp:{mcp_count}"

    def _print_header(
        self,
        *,
        agent_provider: str,
        agent_model_name: str,
        memory_enabled: bool,
        trace_enabled: bool,
        live_events_enabled: bool,
        skill_registry: SkillRegistry | None,
        mcp_result: Any | None,
        mcp_tools: list[BaseTool],
        cwd: Path,
    ) -> None:
        title = self.style.bold(CLI_DISPLAY_NAME)
        title_line = f"{title} {self.style.dim('v' + cli_version())}"
        model_line = self.style.dim(
            f"{agent_provider}:{agent_model_name} · API Usage · "
            f"{self._status('memory', memory_enabled)} · "
            f"{self._status('live', live_events_enabled)} · "
            f"{self._status('trace', trace_enabled)}"
        )
        tool_line = self.style.dim(
            f"{self._short_tool_summary(skill_registry=skill_registry, mcp_result=mcp_result, mcp_tools=mcp_tools)}"
        )
        cwd_line = self.style.dim(format_cli_path(cwd))
        header_lines = [title_line, model_line, tool_line, cwd_line]

        self._print()

        _ansi_re = r"\033\[[0-9;]*m"
        _max_text_w = max(
            (len(re.sub(_ansi_re, "", line)) for line in header_lines),
            default=0,
        )
        _logo_total = 14  # 12 char logo + 2 space gap

        if self.width >= _logo_total + _max_text_w:
            # Wide terminal: side-by-side layout (original behavior)
            for index, (logo_line, logo_color) in enumerate(CLI_LOGO):
                logo = self.style.logo(logo_line, logo_color)
                suffix = header_lines[index] if index < len(header_lines) else ""
                self._print(f"{logo}  {suffix}".rstrip())
        else:
            # Narrow terminal: stacked layout — centered logo on top, text below
            for logo_line, logo_color in CLI_LOGO:
                self._print(self._visible_center(self.style.logo(logo_line, logo_color), self.width))
            self._print()
            for line in header_lines:
                self._print(self._visible_center(line, self.width))

        self._print()
        self._print(self._divider())
        self._print(self.style.dim("? for shortcuts · exit / quit / 退出 to end"))

    def _print_config_section(
        self,
        title: str,
        pairs: list[tuple[str, Any]],
    ) -> None:
        lines = self._render_pairs(pairs)
        if not lines:
            return
        self._print()
        self._print(self.style.dim(title))
        for line in lines:
            self._print(self.style.dim(line))

    def print_startup(
        self,
        *,
        show_config: bool,
        agent_provider: str,
        agent_model_name: str,
        embedding_provider: str,
        embedding_model_name: str,
        actual_query_rewrite_mode: str,
        rewrite_provider: str,
        rewrite_model_name: str,
        bm25_enabled: bool,
        cross_encoder_enabled: bool,
        reranker_provider: str,
        reranker_model_name: str,
        data_dir: str,
        memory_dir: str,
        memory_enabled: bool,
        memory_provider: str,
        memory_model_name: str,
        retry_policy: LLMRetryPolicy,
        max_tool_rounds: int,
        max_repeated_tool_calls: int,
        reflection_enabled: bool,
        stream_output_enabled: bool,
        cache_enabled: bool,
        cache_connected: bool,
        trace_enabled: bool,
        trace_path: Path | None,
        live_events_enabled: bool,
        skill_registry: SkillRegistry | None,
        mcp_result: Any | None,
        mcp_tools: list[BaseTool],
        user_id: str,
    ) -> None:
        self._print_header(
            agent_provider=agent_provider,
            agent_model_name=agent_model_name,
            memory_enabled=memory_enabled,
            trace_enabled=trace_enabled,
            live_events_enabled=live_events_enabled,
            skill_registry=skill_registry,
            mcp_result=mcp_result,
            mcp_tools=mcp_tools,
            cwd=Path.cwd(),
        )
        if not show_config:
            return

        normalized_retry = retry_policy.normalized()
        trace_value = (
            f"on ({trace_path})"
            if trace_enabled and trace_path is not None
            else "off"
        )
        cache_value = self._on_off(cache_enabled)
        if cache_connected:
            cache_value += " (connected)"

        sections: list[tuple[str, list[tuple[str, Any]]]] = [
            (
                "Model",
                [
                    ("agent", f"{agent_provider}:{agent_model_name}"),
                    ("embedding", f"{embedding_provider}:{embedding_model_name}"),
                    (
                        "rewrite",
                        (
                            f"{rewrite_provider}:{rewrite_model_name}"
                            if actual_query_rewrite_mode != "off"
                            else None
                        ),
                    ),
                    (
                        "memory",
                        (
                            f"{memory_provider}:{memory_model_name}"
                            if memory_enabled
                            else None
                        ),
                    ),
                ],
            ),
            (
                "Retrieval",
                [
                    ("query rewrite", actual_query_rewrite_mode),
                    ("bm25", self._on_off(bm25_enabled)),
                    ("rerank", self._on_off(cross_encoder_enabled)),
                    (
                        "rerank model",
                        (
                            f"{reranker_provider}:{reranker_model_name}"
                            if cross_encoder_enabled
                            else None
                        ),
                    ),
                ],
            ),
            (
                "Runtime",
                [
                    ("memory", self._on_off(memory_enabled)),
                    ("reflection", self._on_off(reflection_enabled)),
                    ("stream output", self._on_off(stream_output_enabled)),
                    ("cache", cache_value),
                    ("trace", trace_value),
                    ("live events", self._on_off(live_events_enabled)),
                    ("user", user_id),
                    ("data", data_dir),
                    ("memory dir", memory_dir),
                ],
            ),
            (
                "Guardrails",
                [
                    (
                        "llm retry",
                        (
                            f"attempts={normalized_retry.max_attempts}, "
                            f"timeout={normalized_retry.per_attempt_timeout_s}s, "
                            f"backoff={normalized_retry.initial_backoff_s}s"
                        ),
                    ),
                    (
                        "tool loop",
                        (
                            f"max_rounds={max_tool_rounds}, "
                            f"max_repeated={max_repeated_tool_calls}"
                        ),
                    ),
                ],
            ),
        ]

        for title, pairs in sections:
            self._print_config_section(title, pairs)

        if skill_registry is not None:
            skills = skill_registry.list_skills()
            skill_names = ", ".join(skill.name for skill in skills) or "none"
            self._print()
            self._print(self.style.dim("Skills"))
            self._print(self.style.dim(f"  mode       on ({len(skills)})"))
            self._print(self.style.dim(f"  available  {skill_names}"))
            if skill_registry.errors:
                self._print(self.style.warning("  warnings"))
                for error in skill_registry.errors:
                    self._print(self.style.warning(f"    - {error}"))
        else:
            self._print()
            self._print(self.style.dim("Skills"))
            self._print(self.style.dim("  mode  off"))

        if mcp_result is not None:
            server_names = ", ".join(mcp_result.server_names) or "none"
            tool_names = ", ".join(tool.name for tool in mcp_tools) or "none"
            self._print()
            self._print(self.style.dim("MCP"))
            self._print(self.style.dim(f"  mode     on ({len(mcp_tools)})"))
            self._print(self.style.dim(f"  servers  {server_names}"))
            self._print(self.style.dim(f"  tools    {tool_names}"))
        else:
            self._print()
            self._print(self.style.dim("MCP"))
            self._print(self.style.dim("  mode  off"))

    def input_prompt(self) -> str:
        return f"\n{self._divider()}\n{self.style.prompt('> ')}"

    def print_exit(self) -> None:
        self._print(f"\n{self.style.dim('Session ended.')}")

    def print_clear(self) -> None:
        self._print(f"\n{self.style.dim('Session context cleared.')}")

    def print_command_result(self, title: str, body: str) -> None:
        self._print()
        self._print(self.style.bold(title))
        for line in body.splitlines() or [""]:
            self._print(self.style.dim(f"  {line}") if line else "")

    def begin_assistant(self) -> None:
        self._print(self.style.bold("Assistant"))

    def start_thinking(self, text: str = "正在分析问题...", *, newline: bool = False) -> "CLIThinkingIndicator":
        indicator = CLIThinkingIndicator(self, text=text)
        self._thinking_indicator = indicator
        indicator.start(newline=newline)
        return indicator

    def update_thinking(self, text: str) -> None:
        indicator = self._thinking_indicator
        if indicator is not None and indicator.active:
            indicator.update(text)

    def print_assistant_delta(self, text: str) -> None:
        self._print(text, end="", flush=True)

    def end_assistant(self) -> None:
        self._print(flush=True)

    def print_assistant_error(self, text: str) -> None:
        self._print()
        self._print(self.style.error("Error"))
        self._print(self.style.error(f"  {text}"))

    def print_live_event(self, text: str) -> None:
        indicator = self._thinking_indicator
        if indicator is not None and indicator.active and indicator.interactive:
            indicator.clear_line()
            self._print(self.style.dim(text), flush=True)
            indicator.render()
            return
        self._print(self.style.dim(text), flush=True)


class CLIThinkingIndicator:
    """Small same-line thinking animation for interactive terminals."""

    DEFAULT_TEXT = "正在分析问题..."
    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    COLOR_STYLES = (
        CLIStyle.ACCENT,
        CLIStyle.LOGO_GREEN,
        CLIStyle.LOGO_PURPLE,
        CLIStyle.WARNING,
    )

    def __init__(
        self,
        view: CLIView,
        *,
        text: str = DEFAULT_TEXT,
        interval_s: float = 0.35,
    ) -> None:
        self.view = view
        self.text = text.rstrip(".")
        self.interval_s = interval_s
        self.interactive = False
        self.active = False
        self._task: asyncio.Task[None] | None = None
        self._color_index = 0
        self._frame_index = 0

    def start(self, *, newline: bool = False) -> None:
        self.interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.active = True
        if newline:
            self.view._print()
        if not self.interactive:
            self.view._print(self.view.style.apply(self.static_text(), CLIStyle.LOGO_GREEN), flush=True)
            return
        self.render()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self._animate())

    async def _animate(self) -> None:
        try:
            while self.active:
                self._frame_index = (self._frame_index + 1) % len(self.SPINNER_FRAMES)
                if self._frame_index == 0:
                    self._color_index = (
                        self._color_index + 1
                    ) % len(self.COLOR_STYLES)
                self.render()
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            return

    def current_text(self) -> str:
        return f"{self.SPINNER_FRAMES[self._frame_index]} {self.text}"

    def static_text(self) -> str:
        return f"✓ {self.text}"

    def update(self, text: str) -> None:
        normalized = text.rstrip(".")
        if not normalized or normalized == self.text:
            return
        self.text = normalized
        self._frame_index = 0
        if self.interactive:
            self.render()
        else:
            self.view._print(self.view.style.apply(self.static_text(), CLIStyle.LOGO_GREEN), flush=True)

    def clear_line(self) -> None:
        if self.interactive:
            print("\r\033[K", end="", flush=True)

    def render(self) -> None:
        if not self.interactive:
            return
        color = self.COLOR_STYLES[self._color_index]
        frame = self.view.style.apply(self.current_text(), color)
        print(f"\r\033[K{frame}", end="", flush=True)

    async def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self.interactive:
            text = self.view.style.apply(f"✓ {self.text}", CLIStyle.LOGO_GREEN)
            print(f"\r\033[K{text}", end="", flush=True)
        else:
            self.clear_line()
        if self.view._thinking_indicator is self:
            self.view._thinking_indicator = None


MEMORY_LAYER_LABELS = {
    "profile": "用户画像与稳定偏好",
    "episode": "历史事件摘要",
    "procedure": "可复用流程记忆",
}


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
        lines.extend(
            [
                (
                    f"- {item['id'][:8]}  "
                    f"{item.get('memory_layer', 'profile')}/{item['memory_type']}  "
                    f"importance={item['importance']:.2f}"
                ),
                f"  {item['content']}",
            ]
        )
    return "\n".join(lines)


def format_shortcuts_help(
    skill_registry: SkillRegistry | None = None,
) -> str:
    command_table = format_slash_command_table(
        available_slash_command_specs(skill_registry)
    )
    sections = [
        "输入 / 会自动显示可用命令；继续输入前缀会实时缩小候选范围。",
        "clear / /clear    清空当前会话上下文",
        "exit / quit / 退出  结束会话",
        "? / help          显示快捷帮助",
    ]
    if command_table:
        sections.extend(["", command_table])
    return "\n".join(sections)


def handle_shortcuts_command(
    user_input: str,
    view: CLIView | None = None,
    skill_registry: SkillRegistry | None = None,
) -> bool:
    if user_input.strip().casefold() not in {"?", "/help", "help"}:
        return False
    body = format_shortcuts_help(skill_registry)
    if view is not None:
        view.print_command_result("Shortcuts", body)
    else:
        print(f"\n{body}")
    return True


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
    view: CLIView | None = None,
) -> bool:
    command, _, argument = user_input.partition(" ")
    if command not in MEMORY_SLASH_COMMANDS:
        return False

    def emit(title: str, body: str) -> None:
        if view is not None:
            view.print_command_result(title, body)
        else:
            print(f"\n{body}")

    if memory_service is None:
        emit("Memory", "记忆系统当前已关闭。")
        return True

    try:
        if command == "/memory":
            emit("Memory", format_memory_list(memory_service.list_memories(user_id)))
            return True

        if command in {"/remember", "/remember-procedure", "/remember-episode"}:
            content = argument.strip()
            if not content:
                emit("Usage", f"{command} 需要记住的内容")
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
            emit("Memory", f"已记住 {record['id'][:8]}\n{record['content']}")
            return True

        if command == "/forget":
            memory_ref = argument.strip()
            if not memory_ref:
                emit("Usage", "/forget 记忆ID前缀")
                return True
            memory_id, error = resolve_memory_id(memory_service, user_id, memory_ref)
            if error is not None or memory_id is None:
                emit("Memory", str(error))
                return True
            memory_service.forget_memory(memory_id, user_id=user_id)
            emit("Memory", "已删除这条记忆。")
            return True

        if command == "/clear-memory":
            count = memory_service.clear_user_memory(user_id)
            emit("Memory", f"已清空 {count} 条长期记忆。")
            return True

    except Exception as error:
        emit("Memory", f"记忆操作失败：{error!r}")
        return True

    return True


def handle_unknown_slash_command(
    user_input: str,
    *,
    view: CLIView | None = None,
    skill_registry: SkillRegistry | None = None,
) -> bool:
    if not is_potential_slash_command(user_input):
        return False
    command = user_input.strip().split(maxsplit=1)[0]
    if find_slash_command_spec(command, skill_registry=skill_registry) is not None:
        return False

    suggestions = suggest_slash_commands(
        command,
        skill_registry=skill_registry,
        limit=5,
    )
    body = format_slash_command_suggestions(suggestions)
    if view is not None:
        view.print_command_result(f"Unknown command: {command}", body)
    else:
        print(f"\n未知命令：{command}\n{body}")
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


def tool_call_id(tool_call: dict[str, Any], index: int) -> str:
    raw_id = tool_call.get("id")
    return str(raw_id) if raw_id else f"invalid-tool-call-{index}"


def invalid_tool_call_message(
    tool_call: dict[str, Any],
    *,
    index: int,
    reason: str,
) -> dict[str, str]:
    return {
        "role": "tool",
        "content": (
            f"工具调用无效：{reason}。"
            f"本次收到的工具调用: {preview_text(tool_call)}。"
            "请重新生成合法的工具调用，或在无法调用工具时直接回答。"
        ),
        "tool_call_id": tool_call_id(tool_call, index),
    }


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
    stream_output_enabled: bool = DEFAULT_STREAM_OUTPUT_ENABLED,
    output_delta_sink: Callable[[str], Any] | None = None,
    cache: JsonCache | None = None,
    cache_ttls: dict[str, int] | None = None,
    view: CLIView | None = None,
    assistant_container: dict[str, Any] | None = None,
):
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)
    retry_policy = llm_retry_policy or LLMRetryPolicy()
    actual_cache_ttls = cache_ttls or {}
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
            cache=cache,
            cache_ttl_s=actual_cache_ttls.get("query_rewrite_ttl_s", 86400),
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
    if stream_output_enabled and output_delta_sink is not None:
        base_model = model_with_streaming_enabled(base_model)
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

    async def load_memory(state: AgentState) -> AgentState:
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
            layer_top_k = {
                "profile": memory_top_k,
                "episode": max(1, memory_top_k // 2),
                "procedure": max(1, memory_top_k // 2),
            }
            if hasattr(memory_service, "asearch_memory_layers"):
                layered_memories = await memory_service.asearch_memory_layers(
                    user_id,
                    user_message,
                    layer_top_k=layer_top_k,
                )
            else:
                layered_memories = await asyncio.to_thread(
                    memory_service.search_memory_layers,
                    user_id,
                    user_message,
                    layer_top_k=layer_top_k,
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
        should_stream_output = (
            stream_output_enabled
            and output_delta_sink is not None
            and hasattr(model, "astream")
        )
        if should_stream_output:
            chunks: list[AIMessageChunk] = []
            emitted_content = ""
            has_tool_calls = False
            response: Any = None
            try:
                async for chunk in model.astream(model_messages):
                    if not isinstance(chunk, AIMessageChunk):
                        response = chunk
                        break
                    chunks.append(chunk)
                    if chunk.tool_calls or chunk.tool_call_chunks:
                        has_tool_calls = True
                        continue
                    content = coerce_message_content(chunk.content)
                    if not content:
                        continue
                    if has_tool_calls:
                        continue
                    await maybe_emit_output_delta(output_delta_sink, content)
                    emitted_content += content
                else:
                    response = ai_message_from_chunks(chunks)
            except Exception as error:
                if emitted_content:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                if not has_tool_calls:
                    if trace_recorder is not None:
                        trace_recorder.event(
                            "model",
                            "agent.model_stream_fallback_to_invoke",
                            {
                                "error": repr(error),
                                "emitted_chars": len(emitted_content),
                            },
                            level="warning",
                        )
                    response = await ainvoke_with_retry(
                        lambda: model.ainvoke(model_messages),
                        retry_policy=retry_policy,
                        operation="agent.model_ainvoke",
                        on_failure=trace_model_retry_failure,
                    )
                    emitted_content = ""
                elif chunks or emitted_content:
                    raise
                else:
                    response = await ainvoke_with_retry(
                        lambda: model.ainvoke(model_messages),
                        retry_policy=retry_policy,
                        operation="agent.model_ainvoke",
                        on_failure=trace_model_retry_failure,
                    )
                    emitted_content = ""
            if isinstance(response, AIMessage) and not response.tool_calls:
                if not emitted_content:
                    answer = coerce_message_content(response.content)
                    if answer:
                        await maybe_emit_output_delta(output_delta_sink, answer)
        else:
            response = await ainvoke_with_retry(
                lambda: model.ainvoke(model_messages),
                retry_policy=retry_policy,
                operation="agent.model_ainvoke",
                on_failure=trace_model_retry_failure,
            )
        # 显示工具调用前的中间文本（non-streaming 路径）
        if (
            view is not None
            and isinstance(response, AIMessage)
            and response.tool_calls
        ):
            answer = coerce_message_content(response.content).strip()
            if answer:
                view._print()
                view.begin_assistant()
                view.print_assistant_delta(answer)
                view.end_assistant()
                if assistant_container is not None:
                    assistant_container["started"] = True
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
            and not should_stream_output
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

        allowed_names = allowed_tool_names()

        async def run_tool_call(
            tool_call: dict[str, Any],
            index: int,
        ) -> tuple[dict[str, Any], Any]:
            tool_name = str(tool_call.get("name") or "").strip()
            if not tool_name:
                return tool_call, invalid_tool_call_message(
                    tool_call,
                    index=index,
                    reason="工具调用缺少 name",
                )
            if "args" not in tool_call:
                return tool_call, invalid_tool_call_message(
                    tool_call,
                    index=index,
                    reason=f"工具 {tool_name} 缺少 args",
                )
            selected_tool = tool_map.get(tool_name)
            if selected_tool is None:
                return tool_call, f"未知工具: {tool_name}"
            elif (
                allowed_names is not None
                and selected_tool.name not in allowed_names
            ):
                allowed = ", ".join(sorted(allowed_names))
                return tool_call, (
                    f"当前已激活 skill 限制可用工具为: {allowed}。"
                    f"已拒绝调用: {selected_tool.name}"
                )

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
            try:
                tool_result = await selected_tool.ainvoke(tool_call.get("args"))
            except ValidationError as error:
                tool_result = format_tool_validation_error(
                    selected_tool.name,
                    error,
                    tool_call.get("args"),
                )
                if trace_recorder is not None:
                    trace_recorder.event(
                        "tool",
                        "agent.tool_call_validation_error",
                        {
                            "tool_name": selected_tool.name,
                            "tool_category": tool_category(selected_tool.name),
                            "args": tool_call.get("args"),
                            "error": repr(error),
                            "result_preview": preview_text(tool_result),
                        },
                        level="warning",
                    )
            except Exception as error:
                if trace_recorder is not None:
                    trace_recorder.event(
                        "tool",
                        "agent.tool_call_error",
                        {
                            "tool_name": selected_tool.name,
                            "tool_category": tool_category(selected_tool.name),
                            "error": repr(error),
                        },
                        level="error",
                    )
                raise
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
            return tool_call, tool_result

        tool_results = await asyncio.gather(
            *(
                run_tool_call(tool_call, index)
                for index, tool_call in enumerate(last_message.tool_calls)
            )
        )

        tool_messages = []
        for index, (tool_call, tool_result) in enumerate(tool_results):
            if isinstance(tool_result, dict) and tool_result.get("role") == "tool":
                tool_messages.append(tool_result)
                continue
            tool_name = str(tool_call.get("name") or "").strip()
            selected_tool = tool_map.get(tool_name)
            if (
                selected_tool is not None
                and selected_tool.name == "load_skill"
                and isinstance(tool_call.get("args"), dict)
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
                    "tool_call_id": tool_call_id(tool_call, index),
                }
            )
        return {
            "messages": tool_messages,
            "active_skill_names": active_skill_names,
            "tool_round_count": tool_round_count,
            "last_tool_call_signature": current_signature,
            "repeated_tool_call_count": repeated_tool_call_count,
        }

    async def save_memory(state: AgentState) -> AgentState:
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
                existing_memories = await asyncio.to_thread(
                    memory_service.search_memory,
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
            if hasattr(memory_extractor, "aextract"):
                extracted_memories = await memory_extractor.aextract(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    existing_memories=existing_memories,
                )
            else:
                extracted_memories = await asyncio.to_thread(
                    memory_extractor.extract,
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
                await asyncio.to_thread(
                    memory_service.add_memories,
                    user_id,
                    new_memories,
                )
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
    stream_output_enabled: bool = DEFAULT_STREAM_OUTPUT_ENABLED,
    cache_enabled: bool = False,
    cache_redis_url: str = "redis://localhost:6379/0",
    cache_namespace: str = "rag-server",
    cache_socket_timeout_s: float = 0.2,
    cache_query_rewrite_ttl_s: int = 86400,
    cache_embedding_ttl_s: int = 604800,
    cache_retrieval_ttl_s: int = 3600,
    cache_rerank_ttl_s: int = 86400,
    cache_memory_ttl_s: int = 300,
) -> None:
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)
    llm_retry_policy = LLMRetryPolicy(
        max_attempts=llm_retry_attempts,
        per_attempt_timeout_s=llm_timeout_s,
        initial_backoff_s=llm_retry_backoff_s,
    )
    view = CLIView()
    cache_ttls = {
        "query_rewrite_ttl_s": cache_query_rewrite_ttl_s,
        "embedding_ttl_s": cache_embedding_ttl_s,
        "retrieval_ttl_s": cache_retrieval_ttl_s,
        "rerank_ttl_s": cache_rerank_ttl_s,
        "memory_ttl_s": cache_memory_ttl_s,
    }
    cache = create_redis_cache(
        enabled=cache_enabled,
        redis_url=cache_redis_url,
        namespace=cache_namespace,
        socket_timeout_s=cache_socket_timeout_s,
    )
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
            CLIStatusEventSink(view),
            CLILiveEventPrinter(enabled=live_events_enabled, view=view),
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
                "stream_output_enabled": stream_output_enabled,
                "cache_enabled": cache_enabled,
                "cache_available": cache is not None,
                "cache_namespace": cache_namespace,
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
        cache=cache,
        cache_ttls=cache_ttls,
    )
    memory_service = (
        MemoryService(
            data_dir=memory_dir,
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name,
            embedding_model_kwargs=embedding_model_kwargs,
            trace_recorder=trace_recorder,
            cache=cache,
            cache_ttls=cache_ttls,
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
    output_delta_dispatcher = OutputDeltaDispatcher()
    assistant_container: dict[str, Any] = {"started": False}
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
        stream_output_enabled=stream_output_enabled,
        output_delta_sink=output_delta_dispatcher,
        cache=cache,
        cache_ttls=cache_ttls,
        assistant_container=assistant_container,
    )
    messages: list[BaseMessage] = []

    # 在 print_startup 之前预创建 PromptSession，终端能力检测耗时被打印输出掩盖。
    prompt_toolkit_session: Any | None = None
    if sys.stdin.isatty() and sys.stdout.isatty() and _PromptToolkitSession is not None:
        prompt_toolkit_session = _PromptToolkitSession()

    view.print_startup(
        show_config=show_config,
        agent_provider=agent_provider,
        agent_model_name=agent_model_name,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        actual_query_rewrite_mode=actual_query_rewrite_mode,
        rewrite_provider=rewrite_provider,
        rewrite_model_name=rewrite_model_name,
        bm25_enabled=bm25_enabled,
        cross_encoder_enabled=cross_encoder_enabled,
        reranker_provider=reranker_provider,
        reranker_model_name=reranker_model_name,
        data_dir=data_dir,
        memory_dir=memory_dir,
        memory_enabled=memory_enabled,
        memory_provider=memory_provider,
        memory_model_name=memory_model_name,
        retry_policy=llm_retry_policy,
        max_tool_rounds=max_tool_rounds,
        max_repeated_tool_calls=max_repeated_tool_calls,
        reflection_enabled=reflection_enabled,
        stream_output_enabled=stream_output_enabled,
        cache_enabled=cache_enabled,
        cache_connected=cache is not None,
        trace_enabled=trace_enabled,
        trace_path=trace_recorder.path if trace_recorder is not None else None,
        live_events_enabled=live_events_enabled,
        skill_registry=skill_registry,
        mcp_result=mcp_result,
        mcp_tools=mcp_tools,
        user_id=user_id,
    )
    slash_commands = available_slash_command_specs(skill_registry)
    try:
        input_session = CLIInputSession(
            view=view,
            slash_commands=slash_commands,
            prompt_toolkit_session=prompt_toolkit_session,
        )
        input_session.enable_completion()
        while True:
            user_input = await input_session.prompt_async()
            if is_cli_exit_command(user_input):
                view.print_exit()
                break
            if not user_input:
                continue
            if is_cli_clear_command(user_input):
                messages = []
                view.print_clear()
                if trace_recorder is not None:
                    trace_recorder.event(
                        "agent",
                        "agent.session_context_cleared",
                        {
                            "user_id": user_id,
                        },
                    )
                continue
            if handle_shortcuts_command(user_input, view, skill_registry):
                continue
            if handle_memory_command(user_input, memory_service, user_id, view):
                continue
            if handle_unknown_slash_command(
                user_input,
                view=view,
                skill_registry=skill_registry,
            ):
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
                thinking_indicator = view.start_thinking()
                assistant_started = False
                streamed_content = ""
                final_message = None
                turn_messages: list[BaseMessage] = []
                stream_sink_used = False
                stream_final_message_seen = False
                assistant_container["started"] = False

                async def print_stream_delta(text: str) -> None:
                    nonlocal assistant_started, stream_sink_used
                    if not text:
                        return
                    if not assistant_started:
                        view.update_thinking("正在组织答案...")
                        await thinking_indicator.stop()
                        view.begin_assistant()
                        assistant_started = True
                    view.print_assistant_delta(text)
                    stream_sink_used = True

                def tool_progress_text(tool_names: list[str]) -> str:
                    """Generate a human-readable progress message for the given tool names."""
                    categories = set()
                    for name in tool_names:
                        if name == "search_product_knowledge":
                            categories.add("搜索知识库")
                        elif name in {"load_skill", "read_skill_file"}:
                            categories.add("加载技能")
                        else:
                            categories.add("调用工具")
                    return "正在" + "、".join(sorted(categories)) + "..."

                output_delta_dispatcher.sink = print_stream_delta
                tool_thinking_active = False
                async for event in app.astream(
                    {
                        "messages": input_messages,
                        "user_id": user_id,
                    },
                    stream_mode="updates",
                ):
                    for node_name, node_output in event.items():
                        if not isinstance(node_output, dict):
                            continue
                        node_messages = coerce_graph_messages(
                            node_output.get("messages")
                        )

                        # When tools node finishes, stop the tool thinking indicator
                        if node_name == "tools" and tool_thinking_active:
                            await thinking_indicator.stop()
                            tool_thinking_active = False

                        turn_messages.extend(node_messages)
                        for msg in node_messages:
                            if not isinstance(msg, AIMessage):
                                continue
                            if msg.tool_calls:
                                # Agent decided to call tools — show tool-specific progress
                                names = [
                                    tc.get("name", "")
                                    for tc in msg.tool_calls
                                ]
                                await thinking_indicator.stop()
                                thinking_indicator = view.start_thinking(
                                    tool_progress_text(names),
                                    newline=assistant_started,
                                )
                                tool_thinking_active = True
                                continue
                            content = coerce_message_content(msg.content).strip()
                            if content and content != streamed_content:
                                stream_final_message_seen = True
                                if not assistant_started:
                                    view.update_thinking("正在组织答案...")
                                    await thinking_indicator.stop()
                                    view.begin_assistant()
                                    assistant_started = True
                                new_text = (
                                    ""
                                    if stream_sink_used
                                    else content[len(streamed_content):]
                                )
                                if stream_sink_used:
                                    await thinking_indicator.stop()
                                elif new_text:
                                    view.print_assistant_delta(new_text)
                                streamed_content = content
                                final_message = msg
                if tool_thinking_active:
                    await thinking_indicator.stop()
                    tool_thinking_active = False
                if streamed_content or stream_sink_used:
                    view.end_assistant()
            except Exception as error:
                if "thinking_indicator" in locals():
                    await thinking_indicator.stop()
                if streamed_content or (stream_sink_used and stream_final_message_seen):
                    view.end_assistant()
                    if trace_recorder is not None:
                        trace_recorder.event(
                            "agent",
                            "agent.user_turn_stream_tail_error_ignored",
                            {
                                "user_id": user_id,
                                "error": repr(error),
                                "elapsed_ms": (
                                    time.perf_counter() - turn_start
                                )
                                * 1000,
                            },
                            level="warning",
                        )
                    continue
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
                view.print_assistant_error(
                    format_tongyi_error(model, model_messages, error)
                )
                continue
            finally:
                output_delta_dispatcher.sink = None

            messages = input_messages + turn_messages

            if final_message is None:
                last = messages[-1] if messages else None
                if isinstance(last, AIMessage):
                    final_message = last

            if isinstance(final_message, AIMessage):
                if not streamed_content and not assistant_container.get("started"):
                    if not assistant_started:
                        view.update_thinking("正在组织答案...")
                        await thinking_indicator.stop()
                        view.begin_assistant()
                        assistant_started = True
                    content = coerce_message_content(final_message.content)
                    view.print_assistant_delta(content)
                    view.end_assistant()
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
            else:
                await thinking_indicator.stop()
    finally:
        if "input_session" in locals():
            input_session.restore_completion()
        if memory_service is not None:
            memory_service.close()


def run_cli(**kwargs: Any) -> None:
    asyncio.run(run_cli_async(**kwargs))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ecommerce customer-service CLI.")
    general = parser.add_argument_group("General")
    model = parser.add_argument_group("Model")
    retrieval = parser.add_argument_group("Retrieval")
    memory = parser.add_argument_group("Memory")
    tools = parser.add_argument_group("Tools")
    tracing = parser.add_argument_group("Tracing")
    guardrails = parser.add_argument_group("Guardrails")
    cache = parser.add_argument_group("Cache")

    general.add_argument(
        "--config",
        default=None,
        help=(
            "Optional .toml or .json config file. "
            "RAG_SERVER_CONFIG is also supported."
        ),
    )
    general.add_argument(
        "--data-dir",
        default=None,
        help="RAG index data directory. Overrides config/env when set.",
    )
    general.add_argument(
        "--memory-dir",
        default=None,
        help="Long-term memory data directory. Overrides config/env when set.",
    )
    model.add_argument(
        "--agent-provider",
        "--chat-provider",
        dest="agent_provider",
        default=None,
        help="Provider used by the agent chat model. Built-ins: tongyi, openai, or an import path.",
    )
    model.add_argument(
        "--agent-model",
        "--chat-model",
        dest="agent_model",
        default=None,
        help="Model name used by the agent chat model. Overrides config/env when set.",
    )
    model.add_argument(
        "--agent-model-kwargs",
        "--chat-model-kwargs",
        dest="agent_model_kwargs",
        default=None,
        help="JSON object passed to the agent chat model constructor.",
    )
    retrieval.add_argument(
        "--query-rewrite",
        choices=QUERY_REWRITE_MODES,
        default=None,
        help=(
            "Control retrieval query rewriting. "
            "'on' is an alias for 'multi_query'. Defaults to config value."
        ),
    )
    retrieval.add_argument(
        "--bm25",
        choices=["on", "off"],
        default=None,
        help="Enable or disable BM25 keyword retrieval. Defaults to config value.",
    )
    retrieval.add_argument(
        "--cross-encoder",
        choices=["on", "off"],
        default=None,
        help="Enable or disable CrossEncoder reranking. Defaults to config value.",
    )
    model.add_argument(
        "--embedding-provider",
        default=None,
        help="Provider used by the embedding model. Built-ins: dashscope, openai, or an import path.",
    )
    model.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name. Defaults to config value.",
    )
    model.add_argument(
        "--embedding-model-kwargs",
        default=None,
        help="JSON object passed to the embedding model constructor.",
    )
    model.add_argument(
        "--reranker-provider",
        "--rerank-provider",
        dest="reranker_provider",
        default=None,
        help="Provider used by reranking. Built-ins: cross_encoder or an import path.",
    )
    model.add_argument(
        "--reranker-model",
        "--rerank-model",
        dest="reranker_model",
        default=None,
        help="Reranker model name. Defaults to config value.",
    )
    model.add_argument(
        "--reranker-model-kwargs",
        "--rerank-model-kwargs",
        dest="reranker_model_kwargs",
        default=None,
        help="JSON object passed to the reranker constructor.",
    )
    model.add_argument(
        "--reranker-device",
        "--rerank-device",
        dest="reranker_device",
        default=None,
        help="Optional device for reranker loading, such as cpu, cuda, or mps.",
    )
    retrieval.add_argument(
        "--reranker-batch-size",
        "--rerank-batch-size",
        dest="reranker_batch_size",
        type=int,
        default=None,
        help="Batch size used by reranker predict(). Defaults to config value.",
    )
    model.add_argument(
        "--rewrite-provider",
        default=None,
        help="Provider used by the query rewrite model. Defaults to the agent provider.",
    )
    model.add_argument(
        "--rewrite-model",
        default=None,
        help="Model name used by the query rewriter. Defaults to the agent model.",
    )
    model.add_argument(
        "--rewrite-model-kwargs",
        default=None,
        help="JSON object passed to the query rewrite model constructor.",
    )
    general.add_argument(
        "--user-id",
        default=None,
        help="User id used to scope long-term memories.",
    )
    memory.add_argument(
        "--memory",
        choices=["on", "off"],
        default=None,
        help="Enable or disable long-term memory. Defaults to config value.",
    )
    model.add_argument(
        "--memory-provider",
        default=None,
        help="Provider used by the memory extractor model. Defaults to the agent provider.",
    )
    model.add_argument(
        "--memory-model",
        default=None,
        help="Model name used by the memory extractor. Defaults to the agent model.",
    )
    model.add_argument(
        "--memory-model-kwargs",
        default=None,
        help="JSON object passed to the memory extractor model constructor.",
    )
    memory.add_argument(
        "--memory-top-k",
        type=int,
        default=None,
        help="Number of long-term memories loaded per profile layer.",
    )
    tools.add_argument(
        "--skills",
        choices=["on", "off"],
        default=None,
        help="Enable or disable Anthropic-style skills. Defaults to config value.",
    )
    tools.add_argument(
        "--skills-dir",
        action="append",
        default=None,
        help=(
            "Additional Anthropic-style skills directory. "
            "Can be passed multiple times. Defaults to .claude/skills."
        ),
    )
    tools.add_argument(
        "--mcp",
        choices=["on", "off"],
        default=None,
        help="Enable or disable MCP client tools. Defaults to config value.",
    )
    tools.add_argument(
        "--mcp-config",
        default=None,
        help=(
            "Path to MCP server JSON config. "
            f"Default config value is {DEFAULT_MCP_CONFIG_PATH}."
        ),
    )
    tracing.add_argument(
        "--trace",
        choices=["on", "off"],
        default=None,
        help="Enable or disable JSONL runtime tracing. Defaults to config value.",
    )
    tracing.add_argument(
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
    general.add_argument(
        "--show-config",
        choices=["on", "off"],
        default=None,
        help=(
            "Show the startup configuration summary in the CLI. "
            "Defaults to config value."
        ),
    )
    general.add_argument(
        "--stream-output",
        choices=["on", "off"],
        default=None,
        help=(
            "Stream assistant answer tokens as they are generated. "
            "Defaults to config value."
        ),
    )
    tracing.add_argument(
        "--trace-dir",
        default=None,
        help=f"Directory for JSONL trace files. Defaults to {DEFAULT_TRACE_DIR}.",
    )
    guardrails.add_argument(
        "--llm-retry-attempts",
        type=int,
        default=None,
        help="Maximum attempts for each LLM call. Defaults to 3.",
    )
    guardrails.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help=(
            "Per-attempt LLM timeout in seconds. "
            "Use 0 or a negative value to disable timeout. Defaults to 30."
        ),
    )
    guardrails.add_argument(
        "--llm-retry-backoff",
        type=float,
        default=None,
        help="Initial retry backoff in seconds. Defaults to 1.",
    )
    guardrails.add_argument(
        "--max-tool-rounds",
        type=int,
        default=None,
        help=(
            "Maximum Agent tool-call rounds per user turn. "
            f"Defaults to {DEFAULT_MAX_TOOL_ROUNDS}."
        ),
    )
    guardrails.add_argument(
        "--max-repeated-tool-calls",
        type=int,
        default=None,
        help=(
            "Maximum repeated identical tool-call rounds per user turn. "
            f"Defaults to {DEFAULT_MAX_REPEATED_TOOL_CALLS}."
        ),
    )
    guardrails.add_argument(
        "--reflection",
        choices=["on", "off"],
        default=None,
        help="Enable or disable post-answer reflection and correction.",
    )
    cache.add_argument(
        "--cache",
        choices=["on", "off"],
        default=None,
        help="Enable or disable Redis-backed runtime cache.",
    )
    cache.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL used when runtime cache is enabled.",
    )
    cache.add_argument(
        "--cache-namespace",
        default=None,
        help="Redis key namespace for runtime cache entries.",
    )
    cache.add_argument(
        "--cache-socket-timeout",
        type=float,
        default=None,
        help="Redis connect/read timeout in seconds.",
    )
    cache.add_argument(
        "--cache-query-rewrite-ttl",
        type=int,
        default=None,
        help="TTL in seconds for query rewrite cache entries.",
    )
    cache.add_argument(
        "--cache-embedding-ttl",
        type=int,
        default=None,
        help="TTL in seconds for embedding cache entries.",
    )
    cache.add_argument(
        "--cache-retrieval-ttl",
        type=int,
        default=None,
        help="TTL in seconds for retrieval result cache entries.",
    )
    cache.add_argument(
        "--cache-rerank-ttl",
        type=int,
        default=None,
        help="TTL in seconds for rerank result cache entries.",
    )
    cache.add_argument(
        "--cache-memory-ttl",
        type=int,
        default=None,
        help="TTL in seconds for memory search cache entries.",
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
    if args.stream_output is not None:
        _put_override(
            overrides,
            "cli",
            "stream_output",
            args.stream_output == "on",
        )
    if args.cache is not None:
        _put_override(overrides, "cache", "enabled", args.cache == "on")
    _put_override(overrides, "cache", "redis_url", args.redis_url)
    _put_override(overrides, "cache", "namespace", args.cache_namespace)
    _put_override(
        overrides,
        "cache",
        "socket_timeout_s",
        args.cache_socket_timeout,
    )
    _put_override(
        overrides,
        "cache",
        "query_rewrite_ttl_s",
        args.cache_query_rewrite_ttl,
    )
    _put_override(overrides, "cache", "embedding_ttl_s", args.cache_embedding_ttl)
    _put_override(overrides, "cache", "retrieval_ttl_s", args.cache_retrieval_ttl)
    _put_override(overrides, "cache", "rerank_ttl_s", args.cache_rerank_ttl)
    _put_override(overrides, "cache", "memory_ttl_s", args.cache_memory_ttl)
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

    clear_terminal_startup()
    run_cli(**config.to_runtime_kwargs())


if __name__ == "__main__":
    main()
