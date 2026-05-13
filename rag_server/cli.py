# 确保所有类型注解在运行时以字符串形式延迟求值，兼容前向引用
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

# Pydantic 用于工具参数校验
from pydantic import ValidationError

# 尝试导入 readline，用于命令行 Tab 补全（Windows 下不可用）
try:
    import readline as _readline
except ImportError:  # pragma: no cover - readline is unavailable on Windows.
    _readline = None

# 尝试导入 prompt_toolkit，用于增强的终端交互体验（实时补全菜单、ANSI 渲染）
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

# LangChain 核心消息类型：用于构建 Agent 的对话历史
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

# LangChain 工具装饰器和基类：用于定义 Agent 可调用的工具
from langchain_core.tools import BaseTool, tool

# LangGraph 状态图框架：构建 Agent 的决策流程
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# 项目内部模块导入
from .cache_service import JsonCache, create_redis_cache  # Redis 缓存服务
from .config import (  # 全局配置管理
    DEFAULT_AGENT_MODEL,
    DEFAULT_CLI_CONFIG_OUTPUT_ENABLED,
    DEFAULT_LIVE_EVENTS_ENABLED,
    DEFAULT_MAX_REPEATED_TOOL_CALLS,
    DEFAULT_MAX_TOOL_ROUNDS,
    DEFAULT_MCP_CONFIG_PATH,
    DEFAULT_QUERY_REWRITE_MODE,
    DEFAULT_REFLECTION_ENABLED,
    DEFAULT_STREAM_OUTPUT_ENABLED,
    DEFAULT_USER_ID,
    QUERY_REWRITE_MODES,
    ConfigError,
    load_app_config,
)
from .llm_retry import LLMRetryPolicy, ainvoke_with_retry  # LLM 调用重试策略
from .mcp_service import load_mcp_tools_from_config  # MCP 协议客户端
from .memory_service import LLMMemoryExtractor, MemoryService  # 长期记忆服务
from .model_factory import (  # 模型工厂：统一创建各类模型实例
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_PROVIDER,
    create_chat_model,
)
from .query_rewrite import (  # 查询改写与多查询融合检索
    LLMQueryRewriter,
    asearch_with_query_rewrites,
)
from .rag_service import RAGService  # RAG 知识库检索服务
from .reflection_service import ReflectionAgent, format_retrieval_results  # 回答反思与修正
from .skill_service import SkillRegistry, build_skill_tools  # Anthropic 风格 Skill 系统
from .trace_service import DEFAULT_TRACE_DIR, TraceRecorder, preview_text  # 运行时追踪与日志
from .utils import call_async_fallback, coerce_message_content, load_prompt

# Import display/view components (extracted to cli_view.py) — re-exported for external consumers  # noqa: F401
from .cli_view import (  # noqa: F401
    CLICompleter, CLIInputSession, CLILiveEventPrinter, CLIStyle,
    CLIStatusEventSink, CLIThinkingIndicator, CLIView,
    PromptToolkitSlashCompleter,
    available_slash_command_specs,
    clear_terminal_startup,
    emit_command_result,
    find_slash_command_spec,
    format_cli_live_event,
    format_layered_memory_context,
    format_memory_list,
    format_shortcuts_help,
    format_slash_command_suggestions,
    format_tongyi_error,
    format_tool_validation_error,
    is_cli_clear_command,
    is_cli_exit_command,
    is_potential_slash_command,
    MEMORY_SLASH_COMMANDS,
    prompt_toolkit_slash_matches,
    suggest_slash_commands,
)
class AgentState(TypedDict, total=False):
    """LangGraph Agent 的共享状态字典

    各节点（load_memory、load_skills、agent、tools、save_memory）
    通过读写此状态来协作完成一轮用户交互。
    """

    # 对话消息列表，使用 add_messages 实现消息累积而非覆盖
    messages: Annotated[list[BaseMessage], add_messages]
    # 当前用户 ID，用于记忆隔离
    user_id: str
    # 最新一条用户消息的文本内容
    latest_user_message: str
    # 当前加载的长期记忆文本（格式化后注入 system prompt）
    memory_context: str
    # 当前加载的 Skill 上下文文本（Skill 元数据或已激活 Skill 的完整内容）
    skill_context: str
    # 当前已激活的 Skill 名称列表（用于工具权限控制）
    active_skill_names: list[str]
    # 工具调用累计轮次，用于循环保护
    tool_round_count: int
    # 上一次工具调用的签名（JSON 序列化），用于检测重复调用
    last_tool_call_signature: str
    # 相同工具调用签名的连续重复次数，超过阈值则触发循环保护
    repeated_tool_call_count: int


# ============================================================================
# 查询改写模式转换
# ============================================================================


def normalize_query_rewrite_mode(mode: str) -> str:
    """将用户友好的 on/off 开关转换为内部策略名称。

    'on' 映射为 'multi_query'（多查询融合检索），其他值保持不变。
    """
    return "multi_query" if mode == "on" else mode
async def _rag_search_async(rag: Any, **kwargs: Any) -> list[dict]:
    """异步 RAG 检索包装器：优先使用原生异步方法，否则在线程池中执行"""
    return await call_async_fallback(rag, "asearch", "search", **kwargs)


async def _rewrite_async(
    rewriter: LLMQueryRewriter,
    question: str,
):
    """异步查询改写包装器：优先使用原生异步方法，否则在线程池中执行"""
    return await call_async_fallback(rewriter, "arewrite", "rewrite", question)
# ============================================================================
# 构建带查询改写的 RAG 检索工具
# ============================================================================


def build_retrieval_tool_with_rewrite(
    rag: RAGService,
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    rewriter: LLMQueryRewriter | None = None,
    trace_recorder: TraceRecorder | None = None,
):
    """构建 search_product_knowledge 工具，内部集成查询改写逻辑。

    根据 query_rewrite_mode 决定检索策略：
    - "off": 直接使用原始查询检索
    - "rewrite_only": 用 LLM 改写查询后检索，失败则回退到原始查询
    - "multi_query" (on): 多查询融合检索——原始查询 + 改写变体 + 关联问题
    """
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)

    @tool(description="检索商品知识，返回与用户问题最相关的商品信息片段。")
    async def search_product_knowledge(question: str) -> str:
        """检索商品知识库，返回与用户问题最相关的商品信息片段。"""
        # 构建 trace 负载，记录问题和改写模式
        trace_payload: dict[str, Any] = {
            "question": question,
            "query_rewrite_mode": actual_query_rewrite_mode,
        }
        # 模式 1: rewrite_only —— 只使用改写后的查询检索
        if actual_query_rewrite_mode == "rewrite_only" and rewriter is not None:
            try:
                rewrite_result = await _rewrite_async(rewriter, question)
            except Exception as error:
                # 改写失败时回退到原始查询
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
                # 改写成功，使用改写后的查询检索
                trace_payload["rewritten_query"] = rewrite_result.rewritten_query
                trace_payload["retrieval_queries"] = [rewrite_result.rewritten_query]
                results = await _rag_search_async(
                    rag,
                    query=rewrite_result.rewritten_query,
                    top_k=3,
                    candidate_top_k=10,
                )
        # 模式 2: multi_query —— 多查询融合检索
        elif actual_query_rewrite_mode == "multi_query" and rewriter is not None:
            try:
                rewrite_result = await _rewrite_async(rewriter, question)
            except Exception as error:
                # 改写失败时回退到原始查询
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
                # 构建多查询列表（原始查询 + 改写变体），去重后融合检索
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
        # 模式 3: off —— 直接使用原始查询检索
        else:
            trace_payload["retrieval_queries"] = [question]
            results = await _rag_search_async(
                rag,
                query=question,
                top_k=3,
                candidate_top_k=10,
            )

        # 未检索到结果时直接返回提示
        if not results:
            if trace_recorder is not None:
                trace_recorder.event(
                    "tool",
                    "tool.search_product_knowledge",
                    {**trace_payload, "result_count": 0},
                )
            return "未检索到相关商品知识。"

        output = format_retrieval_results(results)
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


# ============================================================================
# 消息处理辅助函数
# ============================================================================


def latest_human_text(messages: list[BaseMessage]) -> str:
    """从消息列表中提取最后一条用户消息的纯文本内容"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return coerce_message_content(message.content).strip()
    return ""


def latest_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """从消息列表中获取最后一条 AI 消息（用于记忆提取等场景）"""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def coerce_graph_messages(messages: Any) -> list[BaseMessage]:
    """将图输出中的原始消息转换为 LangChain 标准消息列表"""
    if not isinstance(messages, list):
        return []
    return convert_to_messages(messages)


def format_recent_tool_context(
    messages: list[BaseMessage],
    *,
    max_messages: int = 6,
    max_chars: int = 6000,
) -> str:
    """从对话历史中提取最近 N 条工具调用结果，格式化为反思用的证据文本。

    受 max_messages 和 max_chars 双重限制，避免 context window 过大。
    """
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)][-max_messages:]
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
    """替换 AI 消息的内容（用于反思修正后更新回答）"""
    try:
        return message.model_copy(update={"content": content})
    except AttributeError:
        return AIMessage(content=content)


def add_ai_message_chunks(chunks: list[AIMessageChunk]) -> AIMessageChunk | None:
    """合并流式输出的 AI 消息块（通过 + 运算符累加）"""
    merged: AIMessageChunk | None = None
    for chunk in chunks:
        merged = chunk if merged is None else merged + chunk
    return merged


def ai_message_from_chunks(chunks: list[AIMessageChunk]) -> AIMessage:
    """将流式输出的消息块列表合并为一个完整的 AIMessage"""
    merged = add_ai_message_chunks(chunks)
    if merged is None:
        return AIMessage(content="")
    message = message_chunk_to_message(merged)
    if isinstance(message, AIMessage):
        return message
    return AIMessage(content=coerce_message_content(message.content))
# ============================================================================
# 流式输出与模型配置辅助
# ============================================================================


def model_with_streaming_enabled(model: Any) -> Any:
    """确保模型开启了流式输出模式。"""
    if getattr(model, "streaming", None) is True:
        return model
    try:
        return model.model_copy(update={"streaming": True})
    except AttributeError:
        setattr(model, "streaming", True)
    return model


async def maybe_emit_output_delta(
    sink: Callable[[str], Any] | None,
    text: str,
) -> None:
    """安全地将流式输出文本发送到 sink 回调（如果设置了 sink 且有内容）"""
    if sink is None or not text:
        return
    result = sink(text)
    if inspect.isawaitable(result):
        await result


class OutputDeltaDispatcher:
    """流式输出的可替换接收器。

    在每次用户轮次中，run_cli_async 会将 print_stream_delta 绑定为 sink，
    轮次结束后置为 None 防止泄漏。
    """

    def __init__(self) -> None:
        self.sink: Callable[[str], Any] | None = None

    async def __call__(self, text: str) -> None:
        await maybe_emit_output_delta(self.sink, text)
def message_usage_metadata(message: BaseMessage) -> dict[str, Any]:
    """从 AI 消息中提取 token 用量元数据（不依赖特定 provider 的 schema）。

    同时检查 usage_metadata 和 response_metadata 两种来源，
    提取 token_usage、model_name、finish_reason 等信息。
    """
    usage: dict[str, Any] = {}
    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        usage["usage_metadata"] = usage_metadata

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if token_usage is not None:
            usage["token_usage"] = token_usage
        model_name = response_metadata.get("model_name") or response_metadata.get("model")
        if model_name is not None:
            usage["model_name"] = model_name
        finish_reason = response_metadata.get("finish_reason")
        if finish_reason is not None:
            usage["finish_reason"] = finish_reason
    return usage
def handle_shortcuts_command(
    user_input: str,
    view: CLIView | None = None,
    skill_registry: SkillRegistry | None = None,
) -> bool:
    """处理快捷帮助命令（? 或 /help 或 help）。

    返回 True 表示命令已处理，调用方不应继续处理该输入。
    """
    if user_input.strip().casefold() not in {"?", "/help", "help"}:
        return False
    body = format_shortcuts_help(skill_registry)
    emit_command_result(view, "Shortcuts", body)
    return True


def resolve_memory_id(
    memory_service: MemoryService,
    user_id: str,
    memory_ref: str,
) -> tuple[str | None, str | None]:
    """根据记忆 ID 前缀解析出完整 ID。

    精确匹配或前缀匹配；多匹配时返回错误信息。
    返回 (memory_id, error_message) 元组，二者互斥。
    """
    memories = memory_service.list_memories(user_id, limit=200)
    matches = [item["id"] for item in memories if item["id"] == memory_ref or item["id"].startswith(memory_ref)]
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
    """处理记忆管理斜杠命令。

    包括：/memory（查看）、/remember（记录）、/forget（删除）、/clear-memory（清空）。
    返回 True 表示命令已处理。
    """
    command, _, argument = user_input.partition(" ")
    if command not in MEMORY_SLASH_COMMANDS:
        return False

    if memory_service is None:
        emit_command_result(view,"Memory", "记忆系统当前已关闭。")
        return True

    try:
        # /memory：列出当前用户的所有长期记忆
        if command == "/memory":
            emit_command_result(view,"Memory", format_memory_list(memory_service.list_memories(user_id)))
            return True

        # /remember /remember-procedure /remember-episode：手动添加记忆
        if command in {"/remember", "/remember-procedure", "/remember-episode"}:
            content = argument.strip()
            if not content:
                emit_command_result(view,"Usage", f"{command} 需要记住的内容")
                return True
            # 根据命令名映射记忆类型
            memory_type = {
                "/remember": "instruction",
                "/remember-procedure": "procedure",
                "/remember-episode": "episode",
            }[command]
            record = memory_service.add_memory(
                user_id,
                content,
                memory_type=memory_type,
                importance=0.8,  # 手动添加的记忆默认重要性 0.8
                source="manual",  # 标记为手动来源
            )
            emit_command_result(view,"Memory", f"已记住 {record['id'][:8]}\n{record['content']}")
            return True

        # /forget：根据 ID 前缀删除单条记忆
        if command == "/forget":
            memory_ref = argument.strip()
            if not memory_ref:
                emit_command_result(view,"Usage", "/forget 记忆ID前缀")
                return True
            memory_id, error = resolve_memory_id(memory_service, user_id, memory_ref)
            if error is not None or memory_id is None:
                emit_command_result(view,"Memory", str(error))
                return True
            memory_service.forget_memory(memory_id, user_id=user_id)
            emit_command_result(view,"Memory", "已删除这条记忆。")
            return True

        # /clear-memory：清空当前用户的所有记忆
        if command == "/clear-memory":
            count = memory_service.clear_user_memory(user_id)
            emit_command_result(view,"Memory", f"已清空 {count} 条长期记忆。")
            return True

    except Exception as error:
        emit_command_result(view,"Memory", f"记忆操作失败：{error!r}")
        return True

    return True


def handle_unknown_slash_command(
    user_input: str,
    *,
    view: CLIView | None = None,
    skill_registry: SkillRegistry | None = None,
) -> bool:
    """处理未知的斜杠命令：显示错误信息和相似命令建议。

    仅对看起来像斜杠命令、但无法匹配到任何已知命令的输入触发。
    返回 True 表示已处理（调用方应跳过后续处理）。
    """
    if not is_potential_slash_command(user_input):
        return False
    command = user_input.strip().split(maxsplit=1)[0]
    # 如果已匹配到已知命令，则不是未知命令，交由后续处理
    if find_slash_command_spec(command, skill_registry=skill_registry) is not None:
        return False

    # 生成相似命令建议
    suggestions = suggest_slash_commands(
        command,
        skill_registry=skill_registry,
        limit=5,
    )
    body = f"未知命令：{command}\n{format_slash_command_suggestions(suggestions)}"
    emit_command_result(view, f"Unknown command: {command}", body)
    return True


def build_tool_map(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """构建工具名到工具对象的查找映射，重名时立即抛出错误。

    这是工具调度的安全前置检查，防止同名工具导致调用歧义。
    """
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
            f"Duplicate tool names detected. Enable MCP tool_name_prefix or rename tools: {duplicate_names}"
        )
    return tool_map


def tool_call_signature(tool_calls: list[dict[str, Any]]) -> str:
    """生成工具调用的签名（JSON 序列化），用于检测重复调用。

    只比较 tool name 和 args，忽略 id 等噪声字段。
    """
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
    """计算重复工具调用计数：如果签名与上次相同则累加，否则重置为 1。

    用于工具循环保护——当重复次数超过阈值时中断工具循环。
    """
    signature = tool_call_signature(tool_calls)
    if signature and signature == state.get("last_tool_call_signature", ""):
        return int(state.get("repeated_tool_call_count", 0)) + 1
    return 1


def tool_call_id(tool_call: dict[str, Any], index: int) -> str:
    """从工具调用字典中提取或生成 tool_call_id"""
    raw_id = tool_call.get("id")
    return str(raw_id) if raw_id else f"invalid-tool-call-{index}"


def invalid_tool_call_message(
    tool_call: dict[str, Any],
    *,
    index: int,
    reason: str,
) -> dict[str, str]:
    """生成工具调用无效时的反馈消息（返回给 Agent 的 ToolMessage 结构）。

    Agent 看到此消息后会重新生成合法的工具调用或直接回答。
    """
    return {
        "role": "tool",
        "content": (
            f"工具调用无效：{reason}。"
            f"本次收到的工具调用: {preview_text(tool_call)}。"
            "请重新生成合法的工具调用，或在无法调用工具时直接回答。"
        ),
        "tool_call_id": tool_call_id(tool_call, index),
    }


# ============================================================================
# LangGraph Agent 构建器
# build_agent() 是整个 CLI 的核心：创建 LangGraph StateGraph，
# 包含 5 个节点 + 条件路由，构成完整的 Agent 循环流水线。
#
# 图结构（节点名 → 说明）：
#   START → load_memory → load_skills → agent ⇄ tools → save_memory → END
#                                           ↓
#                                       loop_guard → save_memory
#
# 每条用户输入都会沿此流水线执行一轮。
# ============================================================================


# ============================================================================
# Agent 图节点工厂函数
# 每个工厂接收显式依赖，返回一个 LangGraph 节点函数（(state) -> state）。
# 将原本嵌套在 build_agent 内的闭包提取为模块级函数，提高可测试性和可读性。
# ============================================================================


def _create_load_memory_node(
    memory_service: MemoryService | None,
    memory_top_k: int,
    trace_recorder: TraceRecorder | None,
) -> Callable[[AgentState], AgentState]:
    """创建 load_memory 节点：检索当前用户的长期记忆并注入上下文。"""

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
            layered_memories = await call_async_fallback(
                memory_service, "asearch_memory_layers", "search_memory_layers",
                user_id, user_message, layer_top_k=layer_top_k,
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
                    "layer_counts": {layer: len(items) for layer, items in layered_memories.items()},
                },
            )
        return {
            "user_id": user_id,
            "latest_user_message": user_message,
            "memory_context": format_layered_memory_context(layered_memories),
        }

    return load_memory


def _create_load_skills_node(
    actual_skill_registry: SkillRegistry | None,
    trace_recorder: TraceRecorder | None,
) -> Callable[[AgentState], AgentState]:
    """创建 load_skills 节点：加载 Skill 发现提示词和显式调用上下文。"""

    def load_skills(state: AgentState) -> AgentState:
        if actual_skill_registry is None:
            return {"skill_context": ""}

        discovery_prompt = actual_skill_registry.discovery_prompt()
        user_message = latest_human_text(state.get("messages", []))
        explicit_skill_name = actual_skill_registry.explicit_invocation_name(user_message)
        explicit_context = actual_skill_registry.render_explicit_skill_context(user_message)
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
                    "available_skill_names": [skill.name for skill in available_skills],
                    "explicit_skill_name": explicit_skill_name,
                    "skill_context_chars": len(update["skill_context"]),
                },
            )
        return update

    return load_skills


def _create_call_model_node(
    *,
    system_prompt: SystemMessage,
    stream_output_enabled: bool,
    output_delta_sink: Callable[[str], Any] | None,
    model: Any,
    retry_policy: LLMRetryPolicy,
    view: CLIView | None,
    assistant_container: dict[str, Any] | None,
    reflection_agent: Any | None,
    trace_recorder: TraceRecorder | None,
) -> Callable[[AgentState], AgentState]:
    """创建 call_model (agent) 节点：调用绑定了工具的 ChatModel，支持流式/非流式输出和反思修正。"""

    def _trace_model_retry_failure(event: dict[str, Any]) -> None:
        if trace_recorder is None:
            return
        trace_recorder.event(
            "model", "agent.model_retry", event, level="warning" if event.get("will_retry") else "error"
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
        should_stream_output = stream_output_enabled and output_delta_sink is not None and hasattr(model, "astream")
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
                if not has_tool_calls and emitted_content:
                    if trace_recorder is not None:
                        trace_recorder.event(
                            "model",
                            "agent.model_stream_recovered_from_chunks",
                            {
                                "error": repr(error),
                                "emitted_chars": len(emitted_content),
                                "chunk_count": len(chunks),
                            },
                            level="warning",
                        )
                    response = ai_message_from_chunks(chunks) if chunks else None
                elif not has_tool_calls:
                    if emitted_content:
                        sys.stdout.write("\r\033[K")
                        sys.stdout.flush()
                    if trace_recorder is not None:
                        trace_recorder.event(
                            "model",
                            "agent.model_stream_fallback_to_invoke",
                            {"error": repr(error), "emitted_chars": len(emitted_content)},
                            level="warning",
                        )
                    response = await ainvoke_with_retry(
                        lambda: model.ainvoke(model_messages),
                        retry_policy=retry_policy,
                        operation="agent.model_ainvoke",
                        on_failure=_trace_model_retry_failure,
                    )
                    emitted_content = ""
                elif chunks or emitted_content:
                    raise
                else:
                    response = await ainvoke_with_retry(
                        lambda: model.ainvoke(model_messages),
                        retry_policy=retry_policy,
                        operation="agent.model_ainvoke",
                        on_failure=_trace_model_retry_failure,
                    )
                    emitted_content = ""
            if isinstance(response, AIMessage) and not response.tool_calls and not emitted_content:
                answer = coerce_message_content(response.content)
                if answer:
                    await maybe_emit_output_delta(output_delta_sink, answer)
        else:
            response = await ainvoke_with_retry(
                lambda: model.ainvoke(model_messages),
                retry_policy=retry_policy,
                operation="agent.model_ainvoke",
                on_failure=_trace_model_retry_failure,
            )
        if view is not None and isinstance(response, AIMessage) and response.tool_calls:
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
                    "has_tool_calls": bool(isinstance(response, AIMessage) and response.tool_calls),
                    "tool_call_count": (
                        len(response.tool_calls) if isinstance(response, AIMessage) and response.tool_calls else 0
                    ),
                    "content_preview": preview_text(coerce_message_content(response.content)),
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
                user_question=state.get("latest_user_message", "") or latest_human_text(state.get("messages", [])),
                initial_answer=initial_answer,
                evidence_context=format_recent_tool_context(state.get("messages", [])),
                memory_context=state.get("memory_context", ""),
                skill_context=state.get("skill_context", ""),
            )
            if revised_answer != initial_answer:
                response = replace_ai_message_content(response, revised_answer)
        return {"messages": [response]}

    return call_model


def _create_route_tools_node(
    max_tool_rounds: int,
    max_repeated_tool_calls: int,
) -> Callable[[AgentState], str]:
    """创建 route_tools 条件路由：根据 Agent 输出决定下一步（tools/loop_guard/save_memory）。"""

    def route_tools(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            tool_round_count = int(state.get("tool_round_count", 0))
            repeated_tool_calls = next_repeated_tool_call_count(state, last_message.tool_calls)
            if tool_round_count >= max_tool_rounds or repeated_tool_calls > max_repeated_tool_calls:
                return "loop_guard"
            return "tools"
        return "save_memory"

    return route_tools


def _create_stop_tool_loop_node(
    max_tool_rounds: int,
    max_repeated_tool_calls: int,
    trace_recorder: TraceRecorder | None,
) -> Callable[[AgentState], AgentState]:
    """创建 stop_tool_loop 节点：超出工具调用上限时中止循环并注入道歉消息。"""

    def stop_tool_loop(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls if isinstance(last_message, AIMessage) and last_message.tool_calls else []
        tool_round_count = int(state.get("tool_round_count", 0))
        repeated_tool_calls = next_repeated_tool_call_count(state, tool_calls)
        reason = "max_tool_rounds" if tool_round_count >= max_tool_rounds else "repeated_tool_calls"
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

    return stop_tool_loop


_MEMORY_EXTRACTION_KEYWORDS = (
    "我叫",
    "我是",
    "我喜欢",
    "我不喜欢",
    "我讨厌",
    "我偏好",
    "记住",
    "别忘了",
    "我的",
    "我身高",
    "我体重",
    "我尺码",
    "我在",
    "我住在",
    "我经常",
    "我习惯",
    "我需要",
    "我要求",
    "我不想要",
    "我过敏",
    "我的地址",
    "我的电话",
    "我的邮箱",
)


def _create_call_tools_node(
    tool_map: dict[str, BaseTool],
    actual_skill_registry: SkillRegistry | None,
    trace_recorder: TraceRecorder | None,
    tool_category: Callable[[str], str],
) -> Callable[[AgentState], AgentState]:
    """创建 call_tools (tools) 节点：并发执行 Agent 请求的所有工具调用。"""

    async def call_tools(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        active_skill_names = list(state.get("active_skill_names", []))
        current_signature = tool_call_signature(last_message.tool_calls)
        repeated_tool_call_count = next_repeated_tool_call_count(state, last_message.tool_calls)
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

        async def run_tool_call(tool_call: dict[str, Any], index: int) -> tuple[dict[str, Any], Any]:
            tool_name = str(tool_call.get("name") or "").strip()
            if not tool_name:
                return tool_call, invalid_tool_call_message(tool_call, index=index, reason="工具调用缺少 name")
            if "args" not in tool_call:
                return tool_call, invalid_tool_call_message(
                    tool_call, index=index, reason=f"工具 {tool_name} 缺少 args"
                )
            selected_tool = tool_map.get(tool_name)
            if selected_tool is None:
                return tool_call, f"未知工具: {tool_name}"
            elif allowed_names is not None and selected_tool.name not in allowed_names:
                allowed = ", ".join(sorted(allowed_names))
                return tool_call, f"当前已激活 skill 限制可用工具为: {allowed}。已拒绝调用: {selected_tool.name}"

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
                tool_result = format_tool_validation_error(selected_tool.name, error, tool_call.get("args"))
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
            *(run_tool_call(tool_call, index) for index, tool_call in enumerate(last_message.tool_calls))
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
                loaded_skill_name = str(tool_call["args"].get("name") or "").strip().lower()
                if (
                    actual_skill_registry is not None
                    and actual_skill_registry.get_skill(loaded_skill_name) is not None
                    and loaded_skill_name not in active_skill_names
                ):
                    active_skill_names.append(loaded_skill_name)
            tool_messages.append(
                {
                    "role": "tool",
                    "content": (tool_result if isinstance(tool_result, str | list) else str(tool_result)),
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

    return call_tools


def _create_save_memory_node(
    memory_service: MemoryService | None,
    memory_extractor: LLMMemoryExtractor | None,
    trace_recorder: TraceRecorder | None,
    extraction_counter: dict[str, int],
    *,
    extraction_interval: int = 3,
) -> Callable[[AgentState], AgentState]:
    """创建 save_memory 节点：从对话中提取并保存长期记忆（带节流控制）。"""

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
        user_message = state.get("latest_user_message", "").strip() or latest_human_text(messages)
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

        turn = extraction_counter.get(user_id, 0)
        extraction_counter[user_id] = turn + 1
        has_personal_info = any(kw in user_message for kw in _MEMORY_EXTRACTION_KEYWORDS)
        should_extract = has_personal_info or turn % extraction_interval == 0

        if not should_extract:
            if trace_recorder is not None:
                trace_recorder.event(
                    "memory",
                    "agent.save_memory_throttled",
                    {"user_id": user_id, "turn": turn + 1, "reason": "interval_not_reached"},
                )
            return {}

        try:
            try:
                existing_memories = await asyncio.to_thread(
                    memory_service.search_memory, user_id, user_message, top_k=8
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
            extracted_memories = await call_async_fallback(
                memory_extractor, "aextract", "extract",
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

            known_contents = {item["content"] for item in memory_service.list_memories(user_id, limit=200)}
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
                await asyncio.to_thread(memory_service.add_memories, user_id, new_memories)
                if trace_recorder is not None:
                    trace_recorder.event(
                        "memory",
                        "agent.save_memory",
                        {
                            "user_id": user_id,
                            "new_memory_count": len(new_memories),
                            "memory_types": [item["memory_type"] for item in new_memories],
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

    return save_memory


def _create_tool_category(mcp_tool_names: set[str]) -> Callable[[str], str]:
    """创建 tool_category 辅助函数：根据工具名判断所属类别（rag/skill/mcp/tool）。"""

    def tool_category(tool_name: str) -> str:
        if tool_name == "search_product_knowledge":
            return "rag"
        if tool_name in {"load_skill", "read_skill_file"}:
            return "skill"
        if tool_name in mcp_tool_names:
            return "mcp"
        return "tool"

    return tool_category


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
    """构建完整的 LangGraph Agent。

    返回值：(compiled_graph, base_model, system_prompt) 三元组。

    内部组装了三个主要部分：
    1. 工具系统：RAG检索工具 + Skill工具 + MCP工具
    2. 模型：绑定所有工具的 ChatModel，按需启用流式输出
    3. 状态图：5 节点 (load_memory, load_skills, agent, tools, save_memory)
       加 loop_guard 和条件路由
    """
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)
    retry_policy = llm_retry_policy or LLMRetryPolicy()
    actual_cache_ttls = cache_ttls or {}
    # 确保循环保护阈值为合法值
    max_tool_rounds = max(0, max_tool_rounds)
    max_repeated_tool_calls = max(1, max_repeated_tool_calls)

    # ----- 初始化查询改写器 -----
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

    # ----- 构建工具集合 -----
    # RAG 检索工具
    retrieval_tool = build_retrieval_tool_with_rewrite(
        rag,
        query_rewrite_mode=actual_query_rewrite_mode,
        rewriter=rewriter,
        trace_recorder=trace_recorder,
    )
    # Skill 工具（load_skill, read_skill_file）
    actual_skill_registry = skill_registry if skills_enabled else None
    skill_tools = (
        build_skill_tools(actual_skill_registry, trace_recorder=trace_recorder)
        if actual_skill_registry is not None
        else []
    )
    # 合并所有工具，构建查找映射
    tools = [retrieval_tool, *skill_tools, *(mcp_tools or [])]
    tool_map = build_tool_map(tools)
    # MCP 工具名集合，用于 tool_category 判断
    mcp_tool_names = {tool.name for tool in (mcp_tools or [])}

    # ----- 初始化 Agent 模型 -----
    base_model = agent_model or create_chat_model(
        provider=agent_provider,
        model_name=agent_model_name,
        **(agent_model_kwargs or {}),
    )
    # 如果启用了流式输出，确保模型支持 astream
    if stream_output_enabled and output_delta_sink is not None:
        base_model = model_with_streaming_enabled(base_model)
    # 用 bind_tools 将工具列表绑定到模型，使其能进行 Function Calling
    model = base_model.bind_tools(tools)

    # ----- 初始化反思修正 Agent -----
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

    # ----- System Prompt：定义 Agent 的角色和行为规则 -----
    system_prompt = SystemMessage(content=load_prompt("system_prompt.txt"))

    # ----- 初始化工具分类器和图节点工厂 -----
    tool_category = _create_tool_category(mcp_tool_names)

    load_memory = _create_load_memory_node(
        memory_service=memory_service,
        memory_top_k=memory_top_k,
        trace_recorder=trace_recorder,
    )
    load_skills = _create_load_skills_node(
        actual_skill_registry=actual_skill_registry,
        trace_recorder=trace_recorder,
    )
    call_model = _create_call_model_node(
        system_prompt=system_prompt,
        stream_output_enabled=stream_output_enabled,
        output_delta_sink=output_delta_sink,
        model=model,
        retry_policy=retry_policy,
        view=view,
        assistant_container=assistant_container,
        reflection_agent=reflection_agent,
        trace_recorder=trace_recorder,
    )
    route_tools = _create_route_tools_node(
        max_tool_rounds=max_tool_rounds,
        max_repeated_tool_calls=max_repeated_tool_calls,
    )
    stop_tool_loop = _create_stop_tool_loop_node(
        max_tool_rounds=max_tool_rounds,
        max_repeated_tool_calls=max_repeated_tool_calls,
        trace_recorder=trace_recorder,
    )
    call_tools = _create_call_tools_node(
        tool_map=tool_map,
        actual_skill_registry=actual_skill_registry,
        trace_recorder=trace_recorder,
        tool_category=tool_category,
    )
    # 记忆提取计数器（跨轮次共享，需要可变容器）
    _memory_extraction_counter: dict[str, int] = {}
    save_memory = _create_save_memory_node(
        memory_service=memory_service,
        memory_extractor=memory_extractor,
        trace_recorder=trace_recorder,
        extraction_counter=_memory_extraction_counter,
    )

    # ----- 组装 LangGraph StateGraph -----
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


# ============================================================================
# 交互式 CLI 主循环
# run_cli_async() 负责：
# 1. 初始化所有服务组件（RAG、Memory、Skills、MCP、Cache、Trace）
# 2. 构建并编译 LangGraph Agent
# 3. 启动交互式输入循环，处理每一条用户输入
# ============================================================================
async def _run_agent_turn(
    *,
    user_input: str,
    messages: list[BaseMessage],
    app: Any,
    view: CLIView,
    user_id: str,
    trace_recorder: TraceRecorder | None,
    output_delta_dispatcher: OutputDeltaDispatcher,
    assistant_container: dict[str, Any],
    system_prompt: SystemMessage,
    model: Any,
) -> list[BaseMessage]:
    """执行单轮 Agent 交互：构造输入、流式执行图、处理输出和异常。

    返回包含本轮新消息的完整消息列表（输入 + 模型回答）。
    """
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

    thinking_indicator = None
    assistant_started = False
    streamed_content = ""
    final_message = None
    turn_messages: list[BaseMessage] = []
    stream_sink_used = False
    stream_final_message_seen = False
    assistant_container["started"] = False
    tool_thinking_active = False

    try:
        thinking_indicator = view.start_thinking()

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
            categories: set[str] = set()
            for name in tool_names:
                if name == "search_product_knowledge":
                    categories.add("搜索知识库")
                elif name in {"load_skill", "read_skill_file"}:
                    categories.add("加载技能")
                else:
                    categories.add("调用工具")
            return "正在" + "、".join(sorted(categories)) + "..."

        output_delta_dispatcher.sink = print_stream_delta
        async for event in app.astream(
            {"messages": input_messages, "user_id": user_id},
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                if not isinstance(node_output, dict):
                    continue
                node_messages = coerce_graph_messages(node_output.get("messages"))

                if node_name == "tools" and tool_thinking_active:
                    await thinking_indicator.stop()
                    tool_thinking_active = False

                turn_messages.extend(node_messages)
                for msg in node_messages:
                    if not isinstance(msg, AIMessage):
                        continue
                    if msg.tool_calls:
                        names = [tc.get("name", "") for tc in msg.tool_calls]
                        await thinking_indicator.stop()
                        thinking_indicator = view.start_thinking(tool_progress_text(names), newline=assistant_started)
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
                        new_text = "" if stream_sink_used else content[len(streamed_content) :]
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
        if thinking_indicator is not None:
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
                        "elapsed_ms": (time.perf_counter() - turn_start) * 1000,
                    },
                    level="warning",
                )
            return input_messages + turn_messages

        if trace_recorder is not None:
            trace_recorder.event(
                "agent",
                "agent.user_turn_error",
                {
                    "user_id": user_id,
                    "error": repr(error),
                    "elapsed_ms": (time.perf_counter() - turn_start) * 1000,
                },
                level="error",
            )
        model_messages = [system_prompt, *input_messages]
        view.print_assistant_error(format_tongyi_error(model, model_messages, error))
        return messages
    finally:
        output_delta_dispatcher.sink = None

    # 合并消息
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
            content = coerce_message_content(final_message.content)
            view.print_assistant_delta(content)
            view.end_assistant()
        if trace_recorder is not None:
            trace_recorder.event(
                "agent",
                "agent.user_turn_end",
                {
                    "user_id": user_id,
                    "elapsed_ms": (time.perf_counter() - turn_start) * 1000,
                    "message_count": len(messages),
                    "output_preview": preview_text(final_message.content),
                    **message_usage_metadata(final_message),
                },
            )
    else:
        await thinking_indicator.stop()

    return messages


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
    # ----- 初始化 TraceRecorder（事件追踪 + 实时日志输出）-----
    # 注册两个 event_sink：
    # 1. CLIStatusEventSink：根据检索/工具事件更新思考动画文本
    # 2. CLILiveEventPrinter：实时打印格式化的 RAG/Memory/Skill/MCP 事件
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
    # 记录启动事件
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
                "llm_retry_backoff_s": (llm_retry_policy.normalized().initial_backoff_s),
                "max_tool_rounds": max_tool_rounds,
                "max_repeated_tool_calls": max_repeated_tool_calls,
                "reflection_enabled": reflection_enabled,
                "stream_output_enabled": stream_output_enabled,
                "cache_enabled": cache_enabled,
                "cache_available": cache is not None,
                "cache_namespace": cache_namespace,
            },
        )
    # ----- 初始化 RAG 服务（FAISS + BM25 + CrossEncoder）-----
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
    # ----- 初始化 MemoryService（SQLite + 按用户隔离的 FAISS 索引）-----
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
    # ----- 初始化 LLMMemoryExtractor（用 LLM 从对话中提取记忆）-----
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
    # ----- 初始化 Skill 注册表（从 .claude/skills/ 发现并加载 Skills）-----
    skill_registry = (
        SkillRegistry.from_project_root(
            Path.cwd(),
            extra_skill_dirs=skill_dirs,
        )
        if skills_enabled
        else None
    )
    # ----- 初始化 MCP 客户端（从配置文件加载远端工具）-----
    mcp_result = await load_mcp_tools_from_config(mcp_config_path) if mcp_enabled else None
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
    # ----- 构建流式输出分发器和助手容器 -----
    # OutputDeltaDispatcher 在每个用户轮次中被绑定为 print_stream_delta，
    # 将模型流式输出的 token 实时打印到终端。
    output_delta_dispatcher = OutputDeltaDispatcher()
    # assistant_container 用于跨轮次追踪"Assistant"标签是否已打印
    assistant_container: dict[str, Any] = {"started": False}
    # ----- 构建 LangGraph Agent -----
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

    # 在 print_startup 之前预创建 PromptSession，
    # 目的是让终端能力检测的耗时发生在打印输出之前，避免用户看到卡顿。
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
        # 创建输入会话并启用斜杠命令补全
        input_session = CLIInputSession(
            view=view,
            slash_commands=slash_commands,
            prompt_toolkit_session=prompt_toolkit_session,
        )
        input_session.enable_completion()

        # ===== 主交互循环 =====
        while True:
            user_input = await input_session.prompt_async()

            # 检查退出命令
            if is_cli_exit_command(user_input):
                view.print_exit()
                break
            # 空输入跳过
            if not user_input:
                continue
            # 清空会话上下文
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
            # 处理快捷帮助命令（? / /help / help）
            if handle_shortcuts_command(user_input, view, skill_registry):
                continue
            # 处理记忆管理命令（/memory / /remember / /forget 等）
            if handle_memory_command(user_input, memory_service, user_id, view):
                continue
            # 处理未知斜杠命令（显示相似命令建议）
            if handle_unknown_slash_command(
                user_input,
                view=view,
                skill_registry=skill_registry,
            ):
                continue

            # ---- 正常对话轮次 ----
            messages = await _run_agent_turn(
                user_input=user_input,
                messages=messages,
                app=app,
                view=view,
                user_id=user_id,
                trace_recorder=trace_recorder,
                output_delta_dispatcher=output_delta_dispatcher,
                assistant_container=assistant_container,
                system_prompt=system_prompt,
                model=model,
            )
    finally:
        # 清理：恢复 readline 补全器配置，关闭 memory_service
        if "input_session" in locals():
            input_session.restore_completion()
        if memory_service is not None:
            memory_service.close()


# 同步入口：包装 run_cli_async 到 asyncio.run()
def run_cli(**kwargs: Any) -> None:
    asyncio.run(run_cli_async(**kwargs))


# ============================================================================
# 命令行参数解析
# 参数分组：General / Model / Retrieval / Memory / Tools / Tracing / Guardrails / Cache
# ============================================================================
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
        help=("Optional .toml or .json config file. RAG_SERVER_CONFIG is also supported."),
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
        help=("Control retrieval query rewriting. 'on' is an alias for 'multi_query'. Defaults to config value."),
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
        help=("Additional Anthropic-style skills directory. Can be passed multiple times. Defaults to .claude/skills."),
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
        help=(f"Path to MCP server JSON config. Default config value is {DEFAULT_MCP_CONFIG_PATH}."),
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
        help=("Show RAG, memory, skill, and MCP call logs live in the CLI. Defaults to config value."),
    )
    general.add_argument(
        "--show-config",
        choices=["on", "off"],
        default=None,
        help=("Show the startup configuration summary in the CLI. Defaults to config value."),
    )
    general.add_argument(
        "--stream-output",
        choices=["on", "off"],
        default=None,
        help=("Stream assistant answer tokens as they are generated. Defaults to config value."),
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
        help=("Per-attempt LLM timeout in seconds. Use 0 or a negative value to disable timeout. Defaults to 30."),
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
        help=(f"Maximum Agent tool-call rounds per user turn. Defaults to {DEFAULT_MAX_TOOL_ROUNDS}."),
    )
    guardrails.add_argument(
        "--max-repeated-tool-calls",
        type=int,
        default=None,
        help=(
            f"Maximum repeated identical tool-call rounds per user turn. Defaults to {DEFAULT_MAX_REPEATED_TOOL_CALLS}."
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
    """解析命令行参数"""
    return build_arg_parser().parse_args(argv)


def _put_override(
    overrides: dict[str, dict[str, Any]],
    section: str,
    key: str,
    value: Any,
) -> None:
    """辅助函数：将非 None 的 CLI 参数值放入覆盖字典的指定 section"""
    if value is None:
        return
    overrides.setdefault(section, {})[key] = value


def build_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """将 argparse 解析结果转换为配置覆盖字典。

    覆盖按 section 分组（paths/agent/retrieval/llm/memory/skills/mcp/trace/cli/cache），
    传递给 load_app_config 以合并 CLI 参数和配置文件/环境变量。
    """
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
        overrides.setdefault("llm", {})["timeout_s"] = args.llm_timeout if args.llm_timeout > 0 else None
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


# ============================================================================
# CLI 主入口
# 流程：解析参数 → 加载配置（含覆盖） → 清屏 → 启动交互式 Agent
# ============================================================================
def main(argv: list[str] | None = None) -> None:
    """CLI 主入口函数。

    1. 解析命令行参数
    2. 加载配置（合并 .toml/.json 配置文件、环境变量和 CLI 覆盖）
    3. 清屏（交互式终端）
    4. 启动异步主循环
    """
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

