# 启用延迟注解求值（PEP 563），允许在类型注解中使用尚未定义的类型
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

# LangChain 基础工具类和 MCP 多服务器客户端适配器
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .utils import coerce_bool

# 默认的 MCP 配置文件路径
DEFAULT_MCP_CONFIG_PATH = "mcp_servers.json"
# 服务器名称校验正则：仅允许字母、数字、下划线和连字符，长度 1-64
SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# 环境变量占位符正则：匹配 ${VAR_NAME} 或 ${VAR_NAME:-默认值} 格式
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
# Supported MCP transport protocols: stdio (本地进程), sse/http/websocket/streamable_http (远程服务)
SUPPORTED_TRANSPORTS = {"stdio", "sse", "websocket", "http", "streamable_http"}


@dataclass(frozen=True)
class MCPConfig:
    """Normalized MCP client configuration loaded from JSON.

    从 JSON 文件加载并校验后的 MCP 客户端配置。
    """

    # connections: 服务器名称 -> 连接参数字典 的映射
    connections: dict[str, dict[str, Any]]
    # 是否在工具名称前添加服务器前缀，用于区分来自不同 MCP 服务器的同名工具
    tool_name_prefix: bool = True


@dataclass(frozen=True)
class MCPToolLoadResult:
    """MCP tools plus small bits of startup metadata for the CLI.

    MCP 工具加载结果，包含工具列表和服务器元数据。
    """

    # 从各 MCP 服务器加载的 LangChain 兼容工具列表
    tools: list[BaseTool]
    # 已成功加载工具的 MCP 服务器名称，按字母排序
    server_names: list[str]


async def load_mcp_tools_from_config(
    config_path: str | Path,
) -> MCPToolLoadResult:
    """Load MCP tools as LangChain tools from a JSON config file.

    从 JSON 配置文件异步加载 MCP 工具，返回 LangChain 兼容的工具列表。
    """
    # 第一步：加载并校验配置文件，得到规范化的连接信息
    config = load_mcp_config(config_path)
    # 如果没有配置任何连接（或所有连接都被禁用），直接返回空结果
    if not config.connections:
        return MCPToolLoadResult(tools=[], server_names=[])

    # 第二步：使用 MultiServerMCPClient 批量连接所有 MCP 服务器
    client = MultiServerMCPClient(
        config.connections,
        tool_name_prefix=config.tool_name_prefix,
    )
    # 第三步：从所有服务器获取工具列表（异步操作）
    tools = await client.get_tools()
    return MCPToolLoadResult(
        tools=tools,
        server_names=sorted(config.connections),
    )


def load_mcp_config(config_path: str | Path) -> MCPConfig:
    """Read and validate MCP server connections from a JSON file.

    读取并校验 MCP 服务器连接的 JSON 配置文件。

    Supported shapes (支持两种 JSON 结构):

    1. {"servers": {"name": {"transport": "stdio", ...}}}
    2. {"name": {"transport": "stdio", ...}}
    """
    # 解析并展开用户路径（如 ~ 和 ~user）
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"MCP config file not found: {path}")

    # 读取并解析 JSON 文件
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid MCP config JSON: {error}") from error

    # 确保顶层是 JSON 对象（而非数组或基本类型）
    if not isinstance(payload, dict):
        raise ValueError("MCP config must be a JSON object")

    # 提取服务器配置：优先读取 "servers" 键
    raw_servers = payload.get("servers")
    if raw_servers is None:
        # 向后兼容：如果没有 "servers" 顶层键，则从根对象中自动识别服务器配置
        # 判断依据：值为 dict 且包含 "transport" 键
        raw_servers = {
            key: value
            for key, value in payload.items()
            if isinstance(value, dict) and "transport" in value
        }
    if not isinstance(raw_servers, dict):
        raise ValueError("MCP config field 'servers' must be an object")

    # 遍历每个服务器配置，进行规范化和校验
    connections: dict[str, dict[str, Any]] = {}
    for server_name, raw_connection in raw_servers.items():
        normalized = _normalize_connection(str(server_name), raw_connection)
        if normalized is not None:
            connections[str(server_name)] = normalized

    # 读取 tool_name_prefix 配置，默认为 True（启用前缀）
    tool_name_prefix = coerce_bool(payload.get("tool_name_prefix"), default=True)
    return MCPConfig(connections=connections, tool_name_prefix=tool_name_prefix)


def _normalize_connection(
    server_name: str,
    raw_connection: Any,
) -> dict[str, Any] | None:
    """规范化并校验单个 MCP 服务器连接配置。

    Returns:
        规范化后的连接参数字典；如果服务器被禁用则返回 None。
    """
    # 校验服务器名称格式：仅允许字母、数字、下划线、连字符，1-64 字符
    if not SERVER_NAME_PATTERN.fullmatch(server_name):
        raise ValueError(
            f"Invalid MCP server name '{server_name}'. "
            "Use letters, numbers, underscores, or hyphens."
        )
    # 连接配置必须是字典类型
    if not isinstance(raw_connection, dict):
        raise ValueError(f"MCP server '{server_name}' config must be an object")
    # 检查 enabled 字段：如果显式设为 false，则跳过该服务器（返回 None）
    if coerce_bool(raw_connection.get("enabled"), default=True) is False:
        return None

    # 展开配置值中的所有环境变量占位符（跳过 "enabled" 字段）
    connection = {
        key: _expand_env_vars(value)
        for key, value in raw_connection.items()
        if key != "enabled"
    }
    # 规范化传输协议名称：将 "streamable-http" 转换为 "streamable_http"
    transport = str(connection.get("transport") or "").strip()
    transport = "streamable_http" if transport == "streamable-http" else transport
    if transport not in SUPPORTED_TRANSPORTS:
        raise ValueError(
            f"MCP server '{server_name}' has unsupported transport: {transport!r}"
        )
    connection["transport"] = transport

    # 根据传输类型执行不同的校验
    if transport == "stdio":
        # stdio 模式：校验命令行和参数
        _validate_stdio_connection(server_name, connection)
    else:
        # 远程模式（sse/http/websocket/streamable_http）：校验 URL 和请求头
        _validate_url_connection(server_name, connection)
        # HTTP 类传输需要将超时字符串转换为 timedelta 对象
        if transport in {"http", "streamable_http"}:
            _coerce_http_timeouts(connection)

    return connection


def _validate_stdio_connection(
    server_name: str,
    connection: dict[str, Any],
) -> None:
    """校验 stdio 传输协议的 MCP 连接配置。

    stdio 模式通过启动本地子进程来运行 MCP 服务器，需要校验：
    - command: 可执行文件路径或命令名
    - args: 命令行参数列表
    - env: 环境变量字典（可选）
    """
    # command 是必需的，不能为空
    command = str(connection.get("command") or "").strip()
    if not command:
        raise ValueError(f"MCP stdio server '{server_name}' requires 'command'")
    connection["command"] = command

    # args 如果提供则必须是列表，所有元素转为字符串
    args = connection.get("args", [])
    if not isinstance(args, list):
        raise ValueError(
            f"MCP stdio server '{server_name}' field 'args' must be a list"
        )
    connection["args"] = [str(item) for item in args]

    # env 如果提供则必须是字典，所有键值转为字符串
    env = connection.get("env")
    if env is not None and not isinstance(env, dict):
        raise ValueError(
            f"MCP stdio server '{server_name}' field 'env' must be an object"
        )
    if isinstance(env, dict):
        connection["env"] = {str(key): str(value) for key, value in env.items()}


def _validate_url_connection(
    server_name: str,
    connection: dict[str, Any],
) -> None:
    """校验基于 URL 的远程 MCP 连接配置（sse/http/websocket/streamable_http）。

    远程模式通过 HTTP/SSE/WebSocket 连接 MCP 服务器，需要校验：
    - url: 服务器地址
    - headers: 自定义 HTTP 头（可选）
    """
    # url 是必需的，不能为空
    url = str(connection.get("url") or "").strip()
    if not url:
        raise ValueError(f"MCP server '{server_name}' requires 'url'")
    connection["url"] = url

    # headers 如果提供则必须是字典类型
    headers = connection.get("headers")
    if headers is not None and not isinstance(headers, dict):
        raise ValueError(
            f"MCP server '{server_name}' field 'headers' must be an object"
        )


def _coerce_http_timeouts(connection: dict[str, Any]) -> None:
    """将 HTTP 类传输的超时数值（秒）转换为 Python timedelta 对象。

    处理 timeout 和 sse_read_timeout 两个字段。
    """
    for key in ("timeout", "sse_read_timeout"):
        value = connection.get(key)
        # 仅当值为纯数字（非 bool）时才转换——bool 是 int 的子类，需排除
        if isinstance(value, int | float) and not isinstance(value, bool):
            connection[key] = timedelta(seconds=float(value))


def _expand_env_vars(value: Any) -> Any:
    """递归展开配置值中的环境变量占位符。

    支持三种类型：
    - 字符串：替换 ${VAR_NAME} 和 ${VAR_NAME:-default}
    - 列表：递归处理每个元素
    - 字典：递归处理每个值
    - 其他类型（数字、布尔值等）：原样返回
    """
    if isinstance(value, str):
        # 对字符串使用正则替换环境变量占位符
        return ENV_VAR_PATTERN.sub(_replace_env_var, value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_env_vars(item) for key, item in value.items()}
    return value


def _replace_env_var(match: re.Match[str]) -> str:
    """替换单个环境变量占位符为实际值。

    支持格式：
    - ${VAR_NAME} — 必须存在，否则报错
    - ${VAR_NAME:-default} — 不存在时使用默认值
    """
    name = match.group(1)       # 环境变量名
    default = match.group(2)    # 默认值（可选，在 :- 之后）
    value = os.environ.get(name)
    if value is not None:
        return value
    if default is not None:
        return default
    # 变量未设置且无默认值时抛出错误
    raise ValueError(f"Environment variable '{name}' is not set")
