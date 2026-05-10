# 确保所有类型注解在运行时以字符串形式延迟求值，兼容前向引用
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
    DEFAULT_CLI_CONFIG_OUTPUT_ENABLED,
    DEFAULT_LIVE_EVENTS_ENABLED,
    DEFAULT_STREAM_OUTPUT_ENABLED,
    ConfigError,
    load_app_config,
)
from .llm_retry import LLMRetryError, LLMRetryPolicy, ainvoke_with_retry  # LLM 调用重试策略
from .memory_service import LLMMemoryExtractor, MemoryService  # 长期记忆服务
from .mcp_service import DEFAULT_MCP_CONFIG_PATH, load_mcp_tools_from_config  # MCP 协议客户端
from .model_factory import (  # 模型工厂：统一创建各类模型实例
    DEFAULT_CHAT_MODEL,
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
from .reflection_service import ReflectionAgent  # 回答反思与修正
from .skill_service import SkillRegistry, build_skill_tools  # Anthropic 风格 Skill 系统
from .trace_service import DEFAULT_TRACE_DIR, TraceRecorder, preview_text  # 运行时追踪与日志
from .utils import coerce_message_content, load_prompt  # 通用工具：安全提取消息文本、加载提示词文件

# ============================================================================
# 全局默认常量
# ============================================================================

# Agent 默认使用对话模型
DEFAULT_AGENT_MODEL = DEFAULT_CHAT_MODEL
# 默认用户 ID，用于隔离不同用户的记忆
DEFAULT_USER_ID = "default_user"
# 默认查询改写模式：on = multi_query（多查询融合）
DEFAULT_QUERY_REWRITE_MODE = "on"
# 查询改写支持的四种模式
QUERY_REWRITE_MODES = ("on", "off", "rewrite_only", "multi_query")
# 工具调用最大轮次，防止 Agent 陷入无限循环
DEFAULT_MAX_TOOL_ROUNDS = 6
# 相同工具调用签名最大重复次数，超过则触发循环保护
DEFAULT_MAX_REPEATED_TOOL_CALLS = 2
# 默认开启回答反思修正
DEFAULT_REFLECTION_ENABLED = True
# CLI 退出命令集合（支持中英文 + 斜杠前缀）
CLI_EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", "退出", "/退出"}
# CLI 清空会话命令集合
CLI_CLEAR_COMMANDS = {"clear", "/clear", "清空", "/清空"}
# CLI 展示名称
CLI_DISPLAY_NAME = "Tulip Agent"
# 包名，用于获取版本号
CLI_PACKAGE_NAME = "rag-server"
# 版本号回退值（当无法通过包元数据获取时使用）
CLI_VERSION_FALLBACK = "0.1.0"
# CLI 启动时的 ASCII Logo，每行对应颜色（purple/green 代表郁金香的两种色调）
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
# 内置记忆管理斜杠命令集合
MEMORY_SLASH_COMMANDS = {
    "/memory",
    "/remember",
    "/remember-procedure",
    "/remember-episode",
    "/forget",
    "/clear-memory",
}


# 斜杠命令规格数据类（不可变）
# 每个斜杠命令都包含名称、用法、描述、分类和来源信息
# 用于统一定义内置命令和 Skill 注册的命令
@dataclass(frozen=True)
class SlashCommandSpec:
    command: str       # 命令名，如 "/help"
    usage: str         # 用法说明，如 "/help"
    description: str   # 命令描述
    category: str      # 命令分类：Session / Memory / Skills
    source: str = "builtin"  # 命令来源：builtin（内置）或 skill（来自 Skill 注册）

    @property
    def completion(self) -> str:
        """返回补全用的字符串，即命令后跟一个空格"""
        return f"{self.command} "


# 内置斜杠命令列表：包含会话管理、帮助和记忆操作命令
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
# 斜杠命令的合法 token 正则：以 / 开头，后跟字母、数字或连字符
SLASH_COMMAND_TOKEN_PATTERN = re.compile(r"^/[A-Za-z0-9][A-Za-z0-9-]*$")


# ============================================================================
# Agent 状态定义
# AgentState 是 LangGraph 状态图中的核心数据结构，在图的各节点间流转
# add_messages 注解确保 messages 字段以追加方式合并，而非覆盖
# ============================================================================

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


def is_cli_exit_command(value: str) -> bool:
    """判断用户输入是否为退出命令（不区分大小写和前导空格）"""
    return value.strip().casefold() in CLI_EXIT_COMMANDS


def is_cli_clear_command(value: str) -> bool:
    """判断用户输入是否为清空会话命令"""
    return value.strip().casefold() in CLI_CLEAR_COMMANDS


def cli_version() -> str:
    """获取 CLI 版本号：优先从包元数据读取，失败则回退到默认值"""
    try:
        return version(CLI_PACKAGE_NAME)
    except PackageNotFoundError:
        return CLI_VERSION_FALLBACK


def should_use_cli_color(stream: Any | None = None) -> bool:
    """判断当前终端是否应该启用 ANSI 颜色输出。

    检查顺序：NO_COLOR 环境变量 → CLICOLOR_FORCE → 是否是 TTY
    """
    if os.getenv("NO_COLOR") is not None:
        return False
    force_color = os.getenv("CLICOLOR_FORCE")
    if force_color and force_color != "0":
        return True
    target = stream if stream is not None else sys.stdout
    return bool(getattr(target, "isatty", lambda: False)())


def clear_terminal_startup(stream: Any | None = None) -> None:
    """启动时清屏（仅在交互式 TTY 终端中执行）"""
    target = stream if stream is not None else sys.stdout
    if not getattr(target, "isatty", lambda: False)():
        return
    os.system("clear")


def terminal_width(default: int = 88) -> int:
    """获取终端宽度，限制在 52-140 列之间，默认 88 列"""
    return max(52, min(shutil.get_terminal_size((default, 24)).columns, 140))


def format_cli_path(path: Path | str) -> str:
    """将路径格式化为用户友好的显示形式（以 ~ 替代 home 目录）"""
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


# ============================================================================
# CLI 样式系统
# ============================================================================

class CLIStyle:
    """ANSI 终端样式管理器。

    所有颜色输出都是可选的——当 enabled=False 时返回纯文本，
    确保自动化测试捕获的输出保持干净可读。
    """

    # ANSI 转义码定义：重置、加粗、调暗、颜色等
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    PROMPT = "\033[1;37m"          # 加粗白色：输入提示符
    MUTED = "\033[38;5;244m"       # 灰色：次要信息
    ACCENT = "\033[38;5;173m"      # 暖橙色：强调色
    LOGO_PURPLE = "\033[38;5;177m" # 淡紫色：Logo 上部分
    LOGO_GREEN = "\033[38;5;113m"  # 浅绿色：Logo 下部分
    WARNING = "\033[38;5;178m"     # 金黄色：警告信息
    ERROR = "\033[38;5;203m"       # 淡红色：错误信息

    def __init__(self, enabled: bool | None = None) -> None:
        # 如果未显式指定，则根据终端能力自动检测
        self.enabled = should_use_cli_color() if enabled is None else enabled

    def apply(self, text: str, *styles: str) -> str:
        """将 ANSI 样式应用到文本上（仅在启用颜色时生效）"""
        if not self.enabled or not text:
            return text
        return f"{''.join(styles)}{text}{self.RESET}"

    def bold(self, text: str) -> str:
        """加粗文本"""
        return self.apply(text, self.BOLD)

    def dim(self, text: str) -> str:
        """调暗文本（用于次要信息）"""
        return self.apply(text, self.MUTED)

    def logo(self, text: str, color: str) -> str:
        """为 Logo 行着色（支持 purple 和 green 两种颜色）"""
        styles = {
            "purple": self.LOGO_PURPLE,
            "green": self.LOGO_GREEN,
        }
        return self.apply(text, styles.get(color, self.ACCENT))

    def warning(self, text: str) -> str:
        """以警告色（金色）显示文本"""
        return self.apply(text, self.WARNING)

    def error(self, text: str) -> str:
        """以错误色（红色）显示文本"""
        return self.apply(text, self.ERROR)

    def prompt(self, text: str) -> str:
        """以提示符样式（加粗白色）显示文本"""
        return self.apply(text, self.PROMPT)


# ============================================================================
# 错误格式化与异步辅助函数
# ============================================================================

def format_tongyi_error(
    model: Any,
    messages: list[BaseMessage],
    error: Exception,
) -> str:
    """把模型异常转换成更可读的 CLI 错误信息。

    区分 LLMRetryError（重试耗尽）和其他 API 错误，
    提取关键字段（status_code、code、message、request_id）供排查。
    """
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
    """异步 RAG 检索包装器：优先使用原生异步方法，否则在线程池中执行"""
    if hasattr(rag, "asearch"):
        return await rag.asearch(**kwargs)
    return await asyncio.to_thread(rag.search, **kwargs)


async def _rewrite_async(
    rewriter: LLMQueryRewriter,
    question: str,
):
    """异步查询改写包装器：优先使用原生异步方法，否则在线程池中执行"""
    if hasattr(rewriter, "arewrite"):
        return await rewriter.arewrite(question)
    return await asyncio.to_thread(rewriter.rewrite, question)


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

        # 将检索结果格式化为结构化文本块，每个片段包含来源和内容
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
    """替换 AI 消息的内容（用于反思修正后更新回答）"""
    try:
        return message.model_copy(update={"content": content})
    except AttributeError:
        return AIMessage(content=content)


def add_ai_message_chunks(chunks: list[AIMessageChunk]) -> AIMessageChunk | None:
    """合并流式输出的 AI 消息块（通过 + 运算符累加）"""
    merged: AIMessageChunk | None = None
    for chunk in chunks:
        if merged is None:
            merged = chunk
        else:
            merged = merged + chunk
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


def format_tool_validation_error(
    tool_name: str,
    error: ValidationError,
    args: Any,
) -> str:
    """将 Pydantic 工具参数校验错误转换为用户友好的中文错误提示。

    对 read_skill_file 提供特殊指导，帮助用户理解正确的调用方式。
    """
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
    """从 AI 消息中提取 token 用量元数据（不依赖特定 provider 的 schema）。

    同时检查 usage_metadata 和 response_metadata 两种来源，
    提取 token_usage、model_name、finish_reason 等信息。
    """
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


# ============================================================================
# 实时事件格式化（RAG / Memory / Skill / MCP / Tool）
# 这些函数将 trace record 转换为 CLI 中实时显示的结构化日志行
# ============================================================================

def _compact_live_value(value: Any, *, max_chars: int = 180) -> str:
    """将任意值压缩为适合终端单行显示的字符串（限制 max_chars 长度）"""
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
    """从 trace record 或 payload 中提取已用毫秒数"""
    elapsed = record.get("elapsed_ms")
    if not isinstance(elapsed, int | float):
        elapsed = payload.get("elapsed_ms")
    return float(elapsed) if isinstance(elapsed, int | float) else None


def _format_live_status(record: dict[str, Any], payload: dict[str, Any]) -> str:
    """格式化实时事件的状态后缀（level + 用时）"""
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
    """构建实时事件的通用格式化块，包含标题、状态和字段列表"""
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
    """格式化 RAG 检索相关的实时事件显示"""
    # 优先从 payload 中提取查询文本（支持多种键名）
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
    """格式化长期记忆相关的实时事件显示"""
    # 提取各层记忆数量，格式化为 "profile:3, episode:2, procedure:1"
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
    """格式化 Skill 调用相关的实时事件显示"""
    # 显式激活的 skill 以 /skillname 格式显示
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
    """格式化 MCP 工具调用相关的实时事件显示"""
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
    """格式化通用工具调用的实时事件显示（按类别区分 RAG/Skill/MCP）"""
    category = str(payload.get("tool_category") or "")
    if category not in {"rag", "skill", "mcp"}:
        return ""
    label = {"rag": "RAG", "skill": "Skill", "mcp": "MCP"}[category]
    # 根据事件名判断是工具调用开始还是结束阶段
    phase = "start" if name.endswith("_start") else "end"
    return _format_live_block(
        f"{label} {payload.get('tool_name', '')} {phase}",
        [
            # 开始时显示参数，结束时显示结果预览
            ("args", payload.get("args") if phase == "start" else None, 160),
            ("result", payload.get("result_preview") if phase == "end" else None, 120),
        ],
        record,
        payload,
    )


def format_cli_live_event(record: dict[str, Any]) -> str:
    """将一条 trace record 转换为适合 CLI 实时显示的单行/多行日志。

    根据事件类型（rag/memory/skill/mcp/tool）分派到对应的格式化函数。
    不匹配的事件返回空字符串，不输出任何内容。
    """
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
    """在交互式 CLI 会话中即时打印选定的 trace 事件。

    作为 TraceRecorder 的 event_sink，每当日志事件触发时被调用。
    当 view 可用时，通过 view.print_live_event 输出（会考虑 thinking indicator 状态）。
    """
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
    """根据实际运行时事件驱动思考状态指示器的更新。

    作为 TraceRecorder 的 event_sink，监听工具调用和检索事件，
    自动将思考动画的文本更新为对应的状态描述（如"正在检索相关知识..."）。
    """
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
        """根据 trace record 的内容返回对应的思考状态文本"""
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        event_type = str(record.get("type") or "")
        name = str(record.get("name") or "")

        # Agent 开始调用 RAG 检索工具
        if (
            event_type == "tool"
            and name == "agent.tool_call_start"
            and payload.get("tool_category") == "rag"
        ):
            return "正在检索相关知识..."

        # RAG 检索工具执行完成
        if (
            event_type == "tool"
            and name == "tool.search_product_knowledge"
        ):
            result_count = payload.get("result_count")
            if isinstance(result_count, int | float) and result_count > 0:
                # 找到相关知识
                return "找到相关知识..."
            if result_count == 0:
                # 未找到，将直接依赖模型知识回答
                return "未找到相关知识，继续直接回答..."

        # 检索事件的结果更新
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


# ============================================================================
# 斜杠命令管理
# ============================================================================

def builtin_slash_command_specs() -> list[SlashCommandSpec]:
    """获取所有内置斜杠命令的规格列表"""
    return list(BUILTIN_SLASH_COMMANDS)


def skill_slash_command_specs(
    skill_registry: SkillRegistry | None,
) -> list[SlashCommandSpec]:
    """从 Skill 注册表中生成 Skill 对应的斜杠命令规格。

    只有标记为 user_invocable 的 Skill 才会被列出。
    """
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
    """获取当前所有可用的斜杠命令（内置 + Skill 注册），按分类和命令名排序"""
    specs = [*builtin_slash_command_specs(), *skill_slash_command_specs(skill_registry)]
    return sorted(specs, key=lambda item: (item.category, item.command))


def is_potential_slash_command(value: str) -> bool:
    """判断用户输入的第一 token 是否像斜杠命令（以 / 开头，符合命名规范）"""
    token = value.strip().split(maxsplit=1)[0] if value.strip() else ""
    return bool(token.startswith("/") and SLASH_COMMAND_TOKEN_PATTERN.fullmatch(token))


def find_slash_command_spec(
    command: str,
    *,
    skill_registry: SkillRegistry | None = None,
) -> SlashCommandSpec | None:
    """精确查找匹配的斜杠命令规格（不区分大小写）"""
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
    """根据用户输入的前缀建议斜杠命令。

    匹配策略：
    1. 仅输入 / 时返回所有命令
    2. 优先按前缀匹配
    3. 无前缀匹配时使用 difflib 模糊匹配
    """
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

    # 模糊匹配：使用 difflib 找相似命令名
    close_names = difflib.get_close_matches(
        prefix,
        [item.command for item in specs],
        n=limit,
        cutoff=0.35,
    )
    spec_by_name = {item.command: item for item in specs}
    return [spec_by_name[name] for name in close_names if name in spec_by_name]


def format_slash_command_table(specs: list[SlashCommandSpec]) -> str:
    """将斜杠命令列表格式化为分组的表格文本（用于 /help 显示）"""
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
    """格式化为斜杠命令建议列表（用于未知命令时的提示）"""
    if not specs:
        return "没有匹配的 slash 命令。输入 /help 查看可用命令。"
    usage_width = min(max(len(item.usage) for item in specs), 30)
    return "\n".join(
        f"{item.usage.ljust(usage_width)}  {item.description}"
        for item in specs
    )


# ============================================================================
# CLI 输入会话与自动补全系统
# ============================================================================

class CLICompleter:
    """基于 readline 的 Tab 补全器，用于斜杠命令的自动补全。

    当用户在行首输入 / 时触发补全，提供匹配的命令列表。
    """

    def __init__(self, specs: list[SlashCommandSpec]) -> None:
        self.specs = specs
        self._matches: list[str] = []

    def complete(self, text: str, state: int) -> str | None:
        """readline 标准的补全回调：state=0 时构建匹配列表，state>0 时逐一返回"""
        if state == 0:
            self._matches = self._build_matches(text)
        try:
            return self._matches[state]
        except IndexError:
            return None

    def _build_matches(self, text: str) -> list[str]:
        """根据当前输入文本构建匹配的补全选项列表"""
        if _readline is not None:
            buffer = _readline.get_line_buffer()
            begin = _readline.get_begidx()
        else:
            buffer = text
            begin = 0
        # 只在行首（光标位置为 0）时才触发补全
        if begin != 0:
            return []
        prefix = (buffer or text).lower()
        if not prefix.startswith("/"):
            return []
        # 返回所有前缀匹配的命令的补全字符串
        return [
            item.completion
            for item in self.specs
            if item.command.startswith(prefix)
        ]


def prompt_toolkit_slash_matches(
    specs: list[SlashCommandSpec],
    text_before_cursor: str,
) -> list[SlashCommandSpec]:
    """找出光标前输入文本匹配的斜杠命令（用于 prompt_toolkit 实时补全菜单）"""
    if not text_before_cursor.startswith("/"):
        return []
    # 如果已输入空格，说明用户已经开始输入参数，不再补全命令名
    if any(character.isspace() for character in text_before_cursor):
        return []
    prefix = text_before_cursor.lower()
    return [item for item in specs if item.command.startswith(prefix)]


class PromptToolkitSlashCompleter(_PTCompleter):
    """prompt_toolkit 的斜杠命令实时补全菜单。

    在用户输入时实时显示匹配的命令列表和描述。
    """

    def __init__(self, specs: list[SlashCommandSpec]) -> None:
        self.specs = specs

    def get_completions(self, document: Any, complete_event: Any):
        if _PTCompletion is None:
            return
        text_before_cursor = str(getattr(document, "text_before_cursor", ""))
        matches = prompt_toolkit_slash_matches(self.specs, text_before_cursor)
        if not matches:
            return

        # 补全替换的起始位置（光标回到行首开始替换）
        start_position = -len(text_before_cursor)
        for item in matches:
            yield _PTCompletion(
                item.completion,
                start_position=start_position,
                display=item.command,
                display_meta=item.description,
            )


class CLIInputSession:
    """交互式输入包装器，可选启用 readline/prompt_toolkit 补全。

    支持两种补全模式：
    1. prompt_toolkit：实时弹出菜单（优先使用）
    2. readline：按 Tab 触发补全（回退方案）

    通过上下文管理器协议自动启用/恢复补全配置。
    """
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
        # 保存补全器的之前状态，以便退出时恢复
        self._previous_completer: Any | None = None
        self._previous_delims: str | None = None
        self._configured = False

    def __enter__(self) -> CLIInputSession:
        """进入上下文时启用补全"""
        self.enable_completion()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """退出上下文时恢复原始补全配置"""
        self.restore_completion()

    def prompt(self) -> str:
        """同步模式下获取用户输入"""
        return input(self.view.input_prompt()).strip()

    async def prompt_async(self) -> str:
        """异步模式下获取用户输入。

        如果支持 live completion（prompt_toolkit），使用带实时补全菜单的异步输入；
        否则回退到同步 input()。
        """
        if self.should_use_live_completion():
            message = self.view.input_prompt()
            if _PTANSI is not None:
                message = _PTANSI(message)
            return (
                await self.prompt_toolkit_session.prompt_async(
                    message,
                    completer=PromptToolkitSlashCompleter(self.slash_commands),
                    complete_while_typing=True,  # 输入时实时显示补全
                    reserve_space_for_menu=min(max(len(self.slash_commands), 3), 8),
                )
            ).strip()
        return await asyncio.to_thread(self.prompt)

    @property
    def prompt_toolkit_session(self) -> Any:
        """惰性创建 prompt_toolkit 会话"""
        if self._prompt_toolkit_session is None:
            self._prompt_toolkit_session = self.prompt_toolkit_session_factory()
        return self._prompt_toolkit_session

    def should_use_live_completion(self) -> bool:
        """判断是否应该使用 prompt_toolkit 实时补全（需要 stdin/stdout 都是 TTY）"""
        if self.prompt_toolkit_session_factory is None:
            return False
        stdin_isatty = getattr(self.stdin, "isatty", lambda: False)
        stdout_isatty = getattr(self.stdout, "isatty", lambda: False)
        return bool(stdin_isatty() and stdout_isatty())

    def enable_completion(self) -> bool:
        """启用斜杠命令补全（优先 prompt_toolkit，回退 readline）"""
        if self.should_use_live_completion():
            return True
        if self.readline is None:
            return False
        try:
            # 保存当前的补全器配置
            self._previous_completer = self.readline.get_completer()
            try:
                self._previous_delims = self.readline.get_completer_delims()
            except (AttributeError, OSError):
                self._previous_delims = None

            # 设置新的补全器（仅针对斜杠命令）
            completer = CLICompleter(self.slash_commands)
            self.readline.set_completer(completer.complete)
            if hasattr(self.readline, "set_completer_delims"):
                # 从分隔符中移除 /，使得 / 可以被 readline 传递给补全器
                delims = self._previous_delims or " \t\n"
                self.readline.set_completer_delims(delims.replace("/", ""))
            self._bind_tab_completion()
        except (AttributeError, OSError):
            return False
        self._configured = True
        return True

    def _bind_tab_completion(self) -> None:
        """绑定 Tab 键到补全功能（兼容 libedit 和 GNU readline 的不同绑定语法）"""
        doc = str(getattr(self.readline, "__doc__", "") or "").lower()
        binding = "bind ^I rl_complete" if "libedit" in doc else "tab: complete"
        try:
            self.readline.parse_and_bind(binding)
        except (AttributeError, OSError):
            return

    def restore_completion(self) -> None:
        """恢复原始的 readline 补全器配置"""
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


# ============================================================================
# CLI 视图层
# ============================================================================

class CLIView:
    """交互式 CLI 的展示层。

    负责所有输出格式化：Logo、配置摘要、输入提示、思考动画、流式输出、
    实时事件打印、错误显示等。所有输出都经 style 装饰，在非 TTY 环境下自动降级为纯文本。
    """
    """Small presentation layer for the interactive CLI."""

    def __init__(
        self,
        *,
        style: CLIStyle | None = None,
        width: int | None = None,
    ) -> None:
        self.style = style or CLIStyle()
        self.width = width or terminal_width()
        # 当前活跃的思考动画指示器（同一时间只有一个）
        self._thinking_indicator: CLIThinkingIndicator | None = None

    def _print(self, text: str = "", *, end: str = "\n", flush: bool = False) -> None:
        """内部打印方法，统一控制输出行为"""
        print(text, end=end, flush=flush)

    @staticmethod
    def _on_off(value: bool) -> str:
        """将布尔值转换为 on/off 字符串"""
        return "on" if value else "off"

    @staticmethod
    def _render_pairs(pairs: list[tuple[str, Any]]) -> list[str]:
        """将键值对列表格式化为对齐的显示行"""
        visible = [(label, value) for label, value in pairs if value is not None]
        if not visible:
            return []
        width = max(len(label) for label, _ in visible)
        return [f"  {label.ljust(width)}  {value}" for label, value in visible]

    def _divider(self) -> str:
        """生成分隔线（根据终端宽度，灰色显示）"""
        return self.style.dim("-" * self.width)

    @staticmethod
    def _visible_center(text: str, width: int) -> str:
        """在指定宽度内居中文本，忽略 ANSI 转义序列的长度"""
        """Center *text* within *width* columns, ignoring ANSI escape sequences."""
        visible = re.sub(r"\033\[[0-9;]*m", "", text)
        pad = max(0, (width - len(visible)) // 2)
        return " " * pad + text

    def _status(self, label: str, enabled: bool, extra: str | None = None) -> str:
        """生成 on/off 状态字符串，如 "memory:on" 或 "trace:on (traces/)" """
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
        """生成工具摘要行，如 "skills:2 · mcp:3" """
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
        """打印 CLI 头部信息：Logo、模型信息、工具摘要、快捷提示。

        布局会根据终端宽度自适应：
        - 宽终端：Logo 左侧 + 信息右侧（并排）
        - 窄终端：Logo 居中在上方 + 信息居中在下方（堆叠）
        """
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
        _logo_total = 14  # Logo 宽度 12 字符 + 2 字符间距

        if self.width >= _logo_total + _max_text_w:
            # 宽终端模式：Logo 和信息并排显示
            for index, (logo_line, logo_color) in enumerate(CLI_LOGO):
                logo = self.style.logo(logo_line, logo_color)
                suffix = header_lines[index] if index < len(header_lines) else ""
                self._print(f"{logo}  {suffix}".rstrip())
        else:
            # 窄终端模式：Logo 和信息上下堆叠显示
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
        """打印配置摘要的一个分组（如 Model、Retrieval、Runtime 等）"""
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
        """打印 CLI 启动信息。

        包括头部（Logo + 模型/工具摘要），以及可选的完整配置摘要。
        配置摘要分为 Model / Retrieval / Runtime / Guardrails 四组，
        外加 Skills 和 MCP 的详细状态。
        """
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
        """生成输入提示符：分隔线 + 加粗白色 '> ' 符号"""
        return f"\n{self._divider()}\n{self.style.prompt('> ')}"

    def print_exit(self) -> None:
        """打印会话结束提示"""
        self._print(f"\n{self.style.dim('Session ended.')}")

    def print_clear(self) -> None:
        """打印会话上下文清空提示"""
        self._print(f"\n{self.style.dim('Session context cleared.')}")

    def print_command_result(self, title: str, body: str) -> None:
        """打印斜杠命令的执行结果（加粗标题 + 灰色主体）"""
        self._print()
        self._print(self.style.bold(title))
        for line in body.splitlines() or [""]:
            self._print(self.style.dim(f"  {line}") if line else "")

    def begin_assistant(self) -> None:
        """标记 AI 助手开始回复（打印加粗的 'Assistant' 标签）"""
        self._print(self.style.bold("Assistant"))

    def start_thinking(self, text: str = "正在分析问题...", *, newline: bool = False) -> "CLIThinkingIndicator":
        """创建并启动思考动画指示器。

        返回值可用于后续 stop() 停止动画。
        """
        indicator = CLIThinkingIndicator(self, text=text)
        self._thinking_indicator = indicator
        indicator.start(newline=newline)
        return indicator

    def update_thinking(self, text: str) -> None:
        """更新当前思考动画的状态文本（如 '正在检索...' -> '正在组织答案...'）"""
        indicator = self._thinking_indicator
        if indicator is not None and indicator.active:
            indicator.update(text)

    def print_assistant_delta(self, text: str) -> None:
        """增量打印助手的流式输出文本（不换行，立即刷新）"""
        self._print(text, end="", flush=True)

    def end_assistant(self) -> None:
        """助手回复完成，刷新输出缓冲区"""
        self._print(flush=True)

    def print_assistant_error(self, text: str) -> None:
        """打印助手调用的错误信息（红色标题 + 错误详情）"""
        self._print()
        self._print(self.style.error("Error"))
        self._print(self.style.error(f"  {text}"))

    def print_live_event(self, text: str) -> None:
        """打印实时事件日志。

        当思考动画活跃时，先清除动画行 -> 打印事件 -> 重新渲染动画；
        否则直接打印事件行。
        """
        indicator = self._thinking_indicator
        if indicator is not None and indicator.active and indicator.interactive:
            indicator.clear_line()
            self._print(self.style.dim(text), flush=True)
            indicator.render()
            return
        self._print(self.style.dim(text), flush=True)


# ============================================================================
# 终端思考动画指示器
# ============================================================================

class CLIThinkingIndicator:
    """终端同行动画指示器。

    在交互式 TTY 中显示旋转 Braille spinner + 状态描述；
    在非 TTY 中降级为静态文本（如 '✓ 正在分析问题'）。

    通过异步 asyncio.Task 驱动：每 0.35s 切换 spinner 帧，
    每完成一轮（10 帧）切换颜色索引，形成渐变色彩效果。
    """
    """Small same-line thinking animation for interactive terminals."""

    DEFAULT_TEXT = "正在分析问题..."
    # Braille spinner 字符集（10 帧顺序循环）
    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    # 颜色轮换序列：每完成一圈 spinner 后切换到下一个颜色
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
        interval_s: float = 0.35,  # 每帧间隔 0.35 秒
    ) -> None:
        self.view = view
        self.text = text.rstrip(".")  # 去掉尾部句号，动画格式化时自行追加
        self.interval_s = interval_s
        self.interactive = False      # 是否为交互式 TTY
        self.active = False           # 动画是否正在运行
        self._task: asyncio.Task[None] | None = None  # 异步动画任务
        self._color_index = 0         # 当前颜色在 COLOR_STYLES 中的索引
        self._frame_index = 0         # 当前帧在 SPINNER_FRAMES 中的索引

    def start(self, *, newline: bool = False) -> None:
        """启动思考动画。

        在交互式 TTY 中创建异步动画任务并开始渲染；
        在非 TTY 中只输出静态 '✓ + 状态文本'。
        """
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
            return  # 无事件循环时无法启动动画
        self._task = loop.create_task(self._animate())

    async def _animate(self) -> None:
        """异步动画循环：每 interval_s 秒推进一帧，同时管理颜色轮换"""
        try:
            while self.active:
                # 推进帧索引
                self._frame_index = (self._frame_index + 1) % len(self.SPINNER_FRAMES)
                # 每完成一轮 spinner（frame_index 回到 0）切换到下一个颜色
                if self._frame_index == 0:
                    self._color_index = (
                        self._color_index + 1
                    ) % len(self.COLOR_STYLES)
                self.render()
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            return

    def current_text(self) -> str:
        """当前动画帧的显示文本：spinner 字符 + 空格 + 状态文本"""
        return f"{self.SPINNER_FRAMES[self._frame_index]} {self.text}"

    def static_text(self) -> str:
        """非交互式环境的静态文本：✓ + 状态文本"""
        return f"✓ {self.text}"

    def update(self, text: str) -> None:
        """更新状态文本并重置帧位置。

        如果新文本与当前相同则忽略。
        在交互模式下重新渲染；在非交互模式下打印静态更新。
        """
        normalized = text.rstrip(".")
        if not normalized or normalized == self.text:
            return
        self.text = normalized
        self._frame_index = 0  # 重置帧索引，从动画开头重新开始
        if self.interactive:
            self.render()
        else:
            self.view._print(self.view.style.apply(self.static_text(), CLIStyle.LOGO_GREEN), flush=True)

    def clear_line(self) -> None:
        """清除当前终端行（\r + 清空到行尾），用于在动画和事件之间切换"""
        if self.interactive:
            print("\r\033[K", end="", flush=True)

    def render(self) -> None:
        """渲染当前动画帧到终端：\r 回到行首，清除整行，输出当前帧"""
        if not self.interactive:
            return
        color = self.COLOR_STYLES[self._color_index]
        frame = self.view.style.apply(self.current_text(), color)
        print(f"\r\033[K{frame}", end="", flush=True)

    async def stop(self) -> None:
        """停止动画，显示最终完成状态（绿色 ✓）。

        取消异步任务，清除该 indicator 对 view 的引用。
        """
        if not self.active:
            return
        self.active = False
        # 取消并等待异步动画任务完成
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self.interactive:
            # 交互式环境：用绿色 ✓ 替换 spinner
            text = self.view.style.apply(f"✓ {self.text}", CLIStyle.LOGO_GREEN)
            print(f"\r\033[K{text}", end="", flush=True)
        else:
            self.clear_line()
        # 清理视图对当前 indicator 的引用
        if self.view._thinking_indicator is self:
            self.view._thinking_indicator = None


# ============================================================================
# 长期记忆格式化与管理
# ============================================================================

# 记忆三层的用户友好标签
MEMORY_LAYER_LABELS = {
    "profile": "用户画像与稳定偏好",       # 用户的身份、风格偏好、长期约束
    "episode": "历史事件摘要",             # 过去对话的重要事件摘要
    "procedure": "可复用流程记忆",          # 用户自定义的可重复执行流程
}


def format_layered_memory_context(layered_memories: dict[str, list[dict]]) -> str:
    """将三层记忆格式化为 LLM 上下文文本。

    按 profile -> episode -> procedure 顺序组织，
    每层包含记忆序号、类型、重要性和内容。
    无记忆的层自动跳过。
    """
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
    """将记忆列表格式化为可读的摘要（用于 /memory 命令输出）"""
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
    """生成快捷帮助文本，包含基本使用说明和所有可用斜杠命令的表格"""
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
    """处理快捷帮助命令（? 或 /help 或 help）。

    返回 True 表示命令已处理，调用方不应继续处理该输入。
    """
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
    """根据记忆 ID 前缀解析出完整 ID。

    精确匹配或前缀匹配；多匹配时返回错误信息。
    返回 (memory_id, error_message) 元组，二者互斥。
    """
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
    """处理记忆管理斜杠命令。

    包括：/memory（查看）、/remember（记录）、/forget（删除）、/clear-memory（清空）。
    返回 True 表示命令已处理。
    """
    command, _, argument = user_input.partition(" ")
    if command not in MEMORY_SLASH_COMMANDS:
        return False

    def emit(title: str, body: str) -> None:
        """内部辅助：通过 view 或直接 print 输出结果"""
        if view is not None:
            view.print_command_result(title, body)
        else:
            print(f"\n{body}")

    if memory_service is None:
        emit("Memory", "记忆系统当前已关闭。")
        return True

    try:
        # /memory：列出当前用户的所有长期记忆
        if command == "/memory":
            emit("Memory", format_memory_list(memory_service.list_memories(user_id)))
            return True

        # /remember /remember-procedure /remember-episode：手动添加记忆
        if command in {"/remember", "/remember-procedure", "/remember-episode"}:
            content = argument.strip()
            if not content:
                emit("Usage", f"{command} 需要记住的内容")
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
                importance=0.8,       # 手动添加的记忆默认重要性 0.8
                source="manual",       # 标记为手动来源
            )
            emit("Memory", f"已记住 {record['id'][:8]}\n{record['content']}")
            return True

        # /forget：根据 ID 前缀删除单条记忆
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

        # /clear-memory：清空当前用户的所有记忆
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
    body = format_slash_command_suggestions(suggestions)
    if view is not None:
        view.print_command_result(f"Unknown command: {command}", body)
    else:
        print(f"\n未知命令：{command}\n{body}")
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
            "Duplicate tool names detected. "
            f"Enable MCP tool_name_prefix or rename tools: {duplicate_names}"
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

    def tool_category(tool_name: str) -> str:
        """判断工具所属类别（rag/skill/mcp/tool），用于实时事件分类显示"""
        if tool_name == "search_product_knowledge":
            return "rag"
        if tool_name in {"load_skill", "read_skill_file"}:
            return "skill"
        if tool_name in mcp_tool_names:
            return "mcp"
        return "tool"

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

    # ========================================================================
    # 图节点 1: load_memory —— 根据用户消息检索长期记忆
    # 按三层 (profile/episode/procedure) 检索，各层有不同的 top_k 配额，
    # 格式化为文本上下文注入后续的 LLM 调用。
    # ========================================================================
    async def load_memory(state: AgentState) -> AgentState:
        """加载当前用户的长期记忆，返回格式化的记忆上下文。

        如果 memory_service 未启用或用户消息为空，直接返回空上下文。
        检索出错时优雅降级为空记忆（不影响主流程）。
        """
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
            # 各层分配不同的配额：profile 全量，episode/procedure 半量
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
            # 记忆检索失败不影响主流程，降级为空记忆
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

    # ========================================================================
    # 图节点 2: load_skills —— 加载 Skill 上下文
    # 生成 discovery_prompt（所有可用 Skill 的元数据摘要），
    # 并检测用户消息中是否显式调用了某个 Skill（如 /sizing-advice）。
    # ========================================================================
    def load_skills(state: AgentState) -> AgentState:
        """加载 Skills 上下文，供 Agent 了解可用技能及其说明。

        如果用户显式调用了某个 Skill，在上下文中注入其完整内容，
        并将该 Skill 加入 active_skill_names 以控制后续工具权限。
        """
        if actual_skill_registry is None:
            return {"skill_context": ""}

        # discovery_prompt: 所有可用 Skill 的元数据摘要（渐进式披露）
        discovery_prompt = actual_skill_registry.discovery_prompt()
        user_message = latest_human_text(state.get("messages", []))
        # 检测用户是否显式调用了某个 Skill
        explicit_skill_name = actual_skill_registry.explicit_invocation_name(
            user_message
        )
        # 如果显式调用了，渲染该 Skill 的完整内容作为上下文
        explicit_context = actual_skill_registry.render_explicit_skill_context(
            user_message
        )
        # 合并两个上下文块
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

    # LLM 重试失败时的 trace 回调
    def trace_model_retry_failure(event: dict[str, Any]) -> None:
        if trace_recorder is None:
            return
        trace_recorder.event(
            "model",
            "agent.model_retry",
            event,
            level="warning" if event.get("will_retry") else "error",
        )

    # ========================================================================
    # 图节点 3: agent (call_model) —— 核心 LLM 调用节点
    # 将 system prompt + memory context + skill context + 对话历史
    # 组装为完整的 prompt，调用绑定了工具的 ChatModel。
    #
    # 支持两种输出路径：
    #   1. 流式输出 (stream): 逐 token 发送到 output_delta_sink
    #   2. 非流式输出 (invoke): 一次性获取完整回答
    # 都配备了 LLMRetryPolicy 重试机制。
    #
    # 如果启用了 reflection，在非流式路径下对最终回答进行反思修正。
    # ========================================================================
    async def call_model(state: AgentState) -> AgentState:
        """调用绑定了工具的 ChatModel，处理工具调用或生成文本回答。

        组装 prompt = system_prompt + memory_context + skill_context + 对话历史。
        """
        # 构建 prompt 消息列表
        prompt_messages: list[BaseMessage] = [system_prompt]
        # 注入长期记忆上下文（如果有）
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
        # 注入 Skill 上下文（如果有）
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
        # 合并 prompt 和对话历史
        model_messages = [*prompt_messages, *state["messages"]]
        # 判断是否使用流式输出
        should_stream_output = (
            stream_output_enabled
            and output_delta_sink is not None
            and hasattr(model, "astream")
        )
        if should_stream_output:
            # ===== 流式输出路径 =====
            # 通过 astream 逐 chunk 读取，非 tool_calls 的文本 token
            # 立即发送到 output_delta_sink 实现实时打印。
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
                    # 检测到 tool_calls chunk 时标记，后续文本 chunk 跳过不发送
                    if chunk.tool_calls or chunk.tool_call_chunks:
                        has_tool_calls = True
                        continue
                    content = coerce_message_content(chunk.content)
                    if not content:
                        continue
                    if has_tool_calls:
                        continue
                    # 发送文本增量到 sink（实时打印）
                    await maybe_emit_output_delta(output_delta_sink, content)
                    emitted_content += content
                else:
                    # 正常完成：合并所有 chunks 为完整 AIMessage
                    response = ai_message_from_chunks(chunks)
            except Exception as error:
                # 流式输出过程中出错——清理已输出的部分内容
                if emitted_content:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                # 无工具调用时，回退到非流式 invoke（带重试）
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
            # ===== 非流式输出路径 =====
            # 一次性调用模型（带重试），适合不需要实时展示的场景
            response = await ainvoke_with_retry(
                lambda: model.ainvoke(model_messages),
                retry_policy=retry_policy,
                operation="agent.model_ainvoke",
                on_failure=trace_model_retry_failure,
            )
        # 显示工具调用前的中间文本（仅 non-streaming 路径，流式路径已通过 sink 输出）
        # 例如 Agent 在发出 tool_calls 前说的 "让我帮你查一下..."
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
        # ---- 回答反思修正（仅非流式路径 + 纯文本回答） ----
        # 让 ReflectionAgent 审视初始回答，结合检索证据和记忆进行修正。
        # 如果修正后的回答与原始回答不同，替换 AIMessage 的内容。
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

    # ========================================================================
    # 条件路由: route_tools —— 根据 Agent 输出决定下一步
    # 三种可能：
    #   "tools"      — Agent 发出了工具调用，且未触发循环保护 → 执行工具
    #   "loop_guard" — 工具调用轮次/重复次数超出限制 → 进入循环保护
    #   "save_memory"— Agent 直接文本回答（无工具调用）→ 进入记忆保存
    # ========================================================================
    def route_tools(state: AgentState) -> str:
        """根据最后一条 AI 消息判断下一步路由。

        检查最后一条消息：
        - 有 tool_calls + 未超限 → "tools" 节点执行工具
        - 有 tool_calls + 超限 → "loop_guard" 中止循环
        - 无 tool_calls（纯文本回答）→ "save_memory" 保存记忆
        """
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

    # ========================================================================
    # 图节点 5: loop_guard (stop_tool_loop) —— 工具循环保护
    # 当 Agent 超出工具调用上限时，中止当前轮次的所有工具调用，
    # 并注入一条道歉消息，引导用户拆分问题或稍后重试。
    # ========================================================================
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

    # ========================================================================
    # 图节点 4: tools (call_tools) —— 并发执行工具调用
    # 对 Agent 发出的所有 tool_calls 并发执行，支持：
    # - 缺失 name/args 的容错处理
    # - 未知工具的拒绝
    # - Skill 白名单限制（当 active_skill 配置了 allowed_tools 时）
    # - load_skill 成功后自动注册到 active_skill_names
    # ========================================================================
    async def call_tools(state: AgentState) -> AgentState:
        """执行 Agent 请求的所有工具调用，返回对应的 ToolMessage 列表。

        同时更新 tool_round_count 和 repeated_tool_call_count 用于循环保护。
        """
        last_message = state["messages"][-1]
        # 复制 active_skill_names（后续可能被 load_skill 扩展）
        active_skill_names = list(state.get("active_skill_names", []))
        current_signature = tool_call_signature(last_message.tool_calls)
        repeated_tool_call_count = next_repeated_tool_call_count(
            state,
            last_message.tool_calls,
        )
        # 工具调用轮次 +1（第一轮为 1）
        tool_round_count = int(state.get("tool_round_count", 0)) + 1

        def allowed_tool_names() -> set[str] | None:
            """返回当前允许调用的工具名称白名单。

            如果 active_skill 配置了 allowed_tools，则限制为白名单；
            load_skill 和 read_skill_file 始终允许（用于加载更多 Skill）。
            """
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
            # 始终允许 load_skill 和 read_skill_file，以便加载更多 Skill
            return names | {"load_skill", "read_skill_file"}

        allowed_names = allowed_tool_names()

        async def run_tool_call(
            tool_call: dict[str, Any],
            index: int,
        ) -> tuple[dict[str, Any], Any]:
            """执行单个工具调用，返回 (原始_tool_call, 执行结果)。

            包含多层安全校验：缺失 name、缺失 args、未知工具、Skill 白名单拒止。
            校验失败时不抛出异常，而是返回错误消息让 Agent 自行纠正。
            """
            tool_name = str(tool_call.get("name") or "").strip()
            if not tool_name:
                return tool_call, invalid_tool_call_message(
                    tool_call,
                    index=index,
                    reason="工具调用缺少 name",
                )
            # 校验 args 存在性
            if "args" not in tool_call:
                return tool_call, invalid_tool_call_message(
                    tool_call,
                    index=index,
                    reason=f"工具 {tool_name} 缺少 args",
                )
            # 查找工具对象
            selected_tool = tool_map.get(tool_name)
            if selected_tool is None:
                return tool_call, f"未知工具: {tool_name}"
            # Skill 白名单拒止检查
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

        # 并发执行所有工具调用（asyncio.gather 等待全部完成）
        tool_results = await asyncio.gather(
            *(
                run_tool_call(tool_call, index)
                for index, tool_call in enumerate(last_message.tool_calls)
            )
        )

        # 汇总工具结果并构建 ToolMessage 列表
        tool_messages = []
        for index, (tool_call, tool_result) in enumerate(tool_results):
            # 已经是 ToolMessage 格式的直接追加（来自 invalid_tool_call_message）
            if isinstance(tool_result, dict) and tool_result.get("role") == "tool":
                tool_messages.append(tool_result)
                continue
            tool_name = str(tool_call.get("name") or "").strip()
            selected_tool = tool_map.get(tool_name)
            # load_skill 成功后自动注册到 active_skill_names
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
            # 构建标准 ToolMessage
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
        # 返回更新后的状态
        return {
            "messages": tool_messages,
            "active_skill_names": active_skill_names,
            "tool_round_count": tool_round_count,
            "last_tool_call_signature": current_signature,
            "repeated_tool_call_count": repeated_tool_call_count,
        }

    # ========================================================================
    # 图节点 6: save_memory —— 从对话中提取并保存长期记忆
    # 在非流式回答完成后，使用 LLMMemoryExtractor 分析本轮对话，
    # 提取有价值的信息，去重后存入 MemoryService。
    # 仅在 memory_service 和 memory_extractor 都启用时执行。
    # ========================================================================
    async def save_memory(state: AgentState) -> AgentState:
        """保存本轮对话中的有价值信息到长期记忆。

        前提条件：有 memory_service、memory_extractor，且有完整的问答对。
        使用 LLM 从 user_message + assistant_message 中提取记忆条目，
        去重后批量写入。
        """
        if memory_service is None or memory_extractor is None:
            return {}

        messages = state.get("messages", [])
        if not messages:
            return {}

        # 获取最终的 AI 回答（纯文本，无待处理工具调用）
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
        # 用户消息和助手消息不能为空
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
            # 先查找相似已有记忆（用于去重和合并）
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
            # 用 LLM 提取新记忆
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

            # 去重：排除内容与已有记忆完全相同的条目
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
            # 批量写入新记忆
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
            # 记忆保存失败不影响主流程，仅记录 trace
            if trace_recorder is not None:
                trace_recorder.event(
                    "memory",
                    "agent.save_memory_failed",
                    {"user_id": user_id, "error": repr(error)},
                    level="error",
                )
            return {}

        return {}

    # ========================================================================
    # 组装 LangGraph StateGraph
    #
    # 图结构：
    #   START → load_memory → load_skills → agent → [条件路由]
    #                                                  ├─ "tools" → tools → agent（循环）
    #                                                  ├─ "loop_guard" → save_memory → END
    #                                                  └─ "save_memory" → save_memory → END
    #
    # 循环机制：当 agent 发出工具调用且未超限时，路由到 tools，
    #          tools 执行完毕后回到 agent，可能再次发起工具调用。
    #          这个 agent ⇄ tools 循环由 tool_round_count 和
    #          repeated_tool_call_count 共同防护。
    # ========================================================================
    graph = StateGraph(AgentState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("load_skills", load_skills)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("loop_guard", stop_tool_loop)
    graph.add_node("save_memory", save_memory)
    # 线性流水线: START → load_memory → load_skills → agent
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "load_skills")
    graph.add_edge("load_skills", "agent")
    # agent 后的条件路由：根据是否有工具调用决定下一步
    graph.add_conditional_edges(
        "agent",
        route_tools,
        {
            "tools": "tools",           # 有工具调用 → 执行工具
            "loop_guard": "loop_guard", # 超出限制 → 中止循环
            "save_memory": "save_memory", # 纯文本回答 → 保存记忆
        },
    )
    # tools 执行后回到 agent，形成循环（可能发起新的工具调用）
    graph.add_edge("tools", "agent")
    # loop_guard 或纯文本回答后进入 save_memory
    graph.add_edge("loop_guard", "save_memory")
    graph.add_edge("save_memory", END)
    # 编译并返回图、模型和 system_prompt
    return graph.compile(), base_model, system_prompt


# ============================================================================
# 交互式 CLI 主循环
# run_cli_async() 负责：
# 1. 初始化所有服务组件（RAG、Memory、Skills、MCP、Cache、Trace）
# 2. 构建并编译 LangGraph Agent
# 3. 启动交互式输入循环，处理每一条用户输入
# ============================================================================
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
            # 将用户输入加到消息历史前
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
                # 启动思考动画（旋转 spinner）
                thinking_indicator = view.start_thinking()
                assistant_started = False       # "Assistant"标签是否已打印
                streamed_content = ""           # 已在终端显示的流式内容
                final_message = None            # 最终 AI 消息
                turn_messages: list[BaseMessage] = []  # 本轮产生的新消息
                stream_sink_used = False        # 流式 sink 是否已被调用
                stream_final_message_seen = False # 是否已看到最终消息
                assistant_container["started"] = False

                # 流式输出的终端回调：将 token 增量打印到终端
                async def print_stream_delta(text: str) -> None:
                    nonlocal assistant_started, stream_sink_used
                    if not text:
                        return
                    if not assistant_started:
                        # 首次收到 token 时停止思考动画，开始打印回答
                        view.update_thinking("正在组织答案...")
                        await thinking_indicator.stop()
                        view.begin_assistant()
                        assistant_started = True
                    view.print_assistant_delta(text)
                    stream_sink_used = True

                def tool_progress_text(tool_names: list[str]) -> str:
                    """根据工具名生成人类可读的进度描述，如 '正在搜索知识库、调用工具...'"""
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

                # 绑定流式输出 sink 到 OutputDeltaDispatcher
                output_delta_dispatcher.sink = print_stream_delta
                tool_thinking_active = False
                # ---- 执行 LangGraph Agent 图（流式模式） ----
                async for event in app.astream(
                    {
                        "messages": input_messages,
                        "user_id": user_id,
                    },
                    stream_mode="updates",  # 每节点完成后输出状态更新
                ):
                    for node_name, node_output in event.items():
                        if not isinstance(node_output, dict):
                            continue
                        node_messages = coerce_graph_messages(
                            node_output.get("messages")
                        )

                        # tools 节点完成时停止工具相关的思考动画
                        if node_name == "tools" and tool_thinking_active:
                            await thinking_indicator.stop()
                            tool_thinking_active = False

                        turn_messages.extend(node_messages)
                        for msg in node_messages:
                            if not isinstance(msg, AIMessage):
                                continue
                            # Agent 决定调用工具 -> 显示工具进度动画
                            if msg.tool_calls:
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
                            # 纯文本输出（回答或过渡文字）
                            content = coerce_message_content(msg.content).strip()
                            if content and content != streamed_content:
                                stream_final_message_seen = True
                                # 首个文本输出时停止思考动画，打印"Assistant"
                                if not assistant_started:
                                    view.update_thinking("正在组织答案...")
                                    await thinking_indicator.stop()
                                    view.begin_assistant()
                                    assistant_started = True
                                # 计算增量文本（避免重复打印）
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
                # 确保工具思考动画已停止
                if tool_thinking_active:
                    await thinking_indicator.stop()
                    tool_thinking_active = False
                # 结束助手输出
                if streamed_content or stream_sink_used:
                    view.end_assistant()
            except Exception as error:
                # ---- 异常处理 ----
                # 确保思考动画停止
                if "thinking_indicator" in locals():
                    await thinking_indicator.stop()
                # 如果已有部分输出，忽略后续错误（部分回答仍有价值）
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
                # 无任何输出的情况下才显示错误
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
                # 每轮结束后清理流式输出 sink，防止泄漏到下一轮
                output_delta_dispatcher.sink = None

            # ---- 轮次后处理 ----
            # 将本轮产生的新消息合并到对话历史
            messages = input_messages + turn_messages

            # 回退：如果流式过程中未设置 final_message，从消息历史中获取
            if final_message is None:
                last = messages[-1] if messages else None
                if isinstance(last, AIMessage):
                    final_message = last

            # 非流式路径下打印最终回答
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
                # 记录轮次结束事件（含 token 用量等元数据）
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
