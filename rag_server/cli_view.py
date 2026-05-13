from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

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
    BaseMessage,
)

# LangChain 工具装饰器和基类：用于定义 Agent 可调用的工具
from langchain_core.tools import BaseTool

# LangGraph 状态图框架：构建 Agent 的决策流程

# 项目内部模块导入
from .config import (  # 全局配置管理
    DEFAULT_LIVE_EVENTS_ENABLED,
)
from .llm_retry import LLMRetryError, LLMRetryPolicy  # LLM 调用重试策略
from .skill_service import SkillRegistry  # Anthropic 风格 Skill 系统
from .trace_service import preview_text  # 运行时追踪与日志

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
    command: str  # 命令名，如 "/help"
    usage: str  # 用法说明，如 "/help"
    description: str  # 命令描述
    category: str  # 命令分类：Session / Memory / Skills
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
    with contextlib.suppress(OSError):
        absolute_path = absolute_path.resolve()

    home = Path.home()
    try:
        relative_to_home = absolute_path.relative_to(home)
    except ValueError:
        return str(absolute_path)
    return f"~/{relative_to_home}"


def emit_command_result(view: Any, title: str, body: str) -> None:
    """通过 CLIView 或直接 print 输出命令结果。"""
    if view is not None:
        view.print_command_result(title, body)
    else:
        print(f"\n{body}")


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
    PROMPT = "\033[1;37m"  # 加粗白色：输入提示符
    MUTED = "\033[38;5;244m"  # 灰色：次要信息
    ACCENT = "\033[38;5;173m"  # 暖橙色：强调色
    LOGO_PURPLE = "\033[38;5;177m"  # 淡紫色：Logo 上部分
    LOGO_GREEN = "\033[38;5;113m"  # 浅绿色：Logo 下部分
    WARNING = "\033[38;5;178m"  # 金黄色：警告信息
    ERROR = "\033[38;5;203m"  # 淡红色：错误信息

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
            "大模型多次尝试后仍未及时响应或暂时不可用。" if error.attempts > 1 else "大模型调用未及时响应或暂时不可用。"
        )
        return f"{prefix} attempts={error.attempts}, last_error={error.last_error!r}"

    status_code = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    message = getattr(error, "message", None)
    request_id = getattr(error, "request_id", None)
    if status_code and code and message:
        return f"大模型调用失败。 status_code={status_code}, code={code}, message={message}, request_id={request_id}"

    return f"大模型调用失败：{error!r}"


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
    query = payload.get("query") or payload.get("question") or payload.get("original_query")
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
    explicit_skill = f"/{payload['explicit_skill_name']}" if payload.get("explicit_skill_name") else None
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
        view: CLIView | None = None,
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

    def __init__(self, view: CLIView) -> None:
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
        if event_type == "tool" and name == "agent.tool_call_start" and payload.get("tool_category") == "rag":
            return "正在检索相关知识..."

        # RAG 检索工具执行完成
        if event_type == "tool" and name == "tool.search_product_knowledge":
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
            if result_count == 0 and (candidate_count is None or candidate_count == 0):
                return "未找到相关知识，继续直接回答..."

        return ""


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

    startswith_matches = [item for item in specs if item.command.startswith(prefix)]
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
    return "\n".join(f"{item.usage.ljust(usage_width)}  {item.description}" for item in specs)


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
        return [item.completion for item in self.specs if item.command.startswith(prefix)]


def prompt_toolkit_slash_matches(
    specs: list[SlashCommandSpec],
    text_before_cursor: str,
) -> list[SlashCommandSpec]:
    """找出光标前输入文本匹配的斜杠命令（用于 prompt_toolkit 实时补全菜单）"""
    if not text_before_cursor.startswith("/"):
        return []
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

    def __init__(
        self,
        *,
        view: CLIView,
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
            if self._previous_delims is not None and hasattr(self.readline, "set_completer_delims"):
                self.readline.set_completer_delims(self._previous_delims)
        except (AttributeError, OSError):
            return


class CLIView:
    """交互式 CLI 的展示层。

    负责所有输出格式化：Logo、配置摘要、输入提示、思考动画、流式输出、
    实时事件打印、错误显示等。所有输出都经 style 装饰，在非 TTY 环境下自动降级为纯文本。
    """

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
        trace_value = f"on ({trace_path})" if trace_enabled and trace_path is not None else "off"
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
                        (f"{rewrite_provider}:{rewrite_model_name}" if actual_query_rewrite_mode != "off" else None),
                    ),
                    (
                        "memory",
                        (f"{memory_provider}:{memory_model_name}" if memory_enabled else None),
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
                        (f"{reranker_provider}:{reranker_model_name}" if cross_encoder_enabled else None),
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
                        (f"max_rounds={max_tool_rounds}, max_repeated={max_repeated_tool_calls}"),
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

    def start_thinking(self, text: str = "正在分析问题...", *, newline: bool = False) -> CLIThinkingIndicator:
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
        self.interactive = False  # 是否为交互式 TTY
        self.active = False  # 动画是否正在运行
        self._task: asyncio.Task[None] | None = None  # 异步动画任务
        self._color_index = 0  # 当前颜色在 COLOR_STYLES 中的索引
        self._frame_index = 0  # 当前帧在 SPINNER_FRAMES 中的索引

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
                    self._color_index = (self._color_index + 1) % len(self.COLOR_STYLES)
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
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self.interactive:
            # 交互式环境：用绿色 ✓ 替换 spinner
            text = self.view.style.apply(f"✓ {self.text}", CLIStyle.LOGO_GREEN)
            print(f"\r\033[K{text}\n", end="", flush=True)
        else:
            self.clear_line()
        # 清理视图对当前 indicator 的引用
        if self.view._thinking_indicator is self:
            self.view._thinking_indicator = None


MEMORY_LAYER_LABELS = {
    "profile": "用户画像与稳定偏好",  # 用户的身份、风格偏好、长期约束
    "episode": "历史事件摘要",  # 过去对话的重要事件摘要
    "procedure": "可复用流程记忆",  # 用户自定义的可重复执行流程
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
    command_table = format_slash_command_table(available_slash_command_specs(skill_registry))
    sections = [
        "输入 / 会自动显示可用命令；继续输入前缀会实时缩小候选范围。",
        "clear / /clear    清空当前会话上下文",
        "exit / quit / 退出  结束会话",
        "? / help          显示快捷帮助",
    ]
    if command_table:
        sections.extend(["", command_table])
    return "\n".join(sections)


