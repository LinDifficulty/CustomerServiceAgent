from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

DEFAULT_MCP_CONFIG_PATH = "mcp_servers.json"
SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
SUPPORTED_TRANSPORTS = {"stdio", "sse", "websocket", "http", "streamable_http"}


@dataclass(frozen=True)
class MCPConfig:
    """Normalized MCP client configuration loaded from JSON."""

    connections: dict[str, dict[str, Any]]
    tool_name_prefix: bool = True


@dataclass(frozen=True)
class MCPToolLoadResult:
    """MCP tools plus small bits of startup metadata for the CLI."""

    tools: list[BaseTool]
    server_names: list[str]


async def load_mcp_tools_from_config(
    config_path: str | Path,
) -> MCPToolLoadResult:
    """Load MCP tools as LangChain tools from a JSON config file."""
    config = load_mcp_config(config_path)
    if not config.connections:
        return MCPToolLoadResult(tools=[], server_names=[])

    client = MultiServerMCPClient(
        config.connections,
        tool_name_prefix=config.tool_name_prefix,
    )
    tools = await client.get_tools()
    return MCPToolLoadResult(
        tools=tools,
        server_names=sorted(config.connections),
    )


def load_mcp_config(config_path: str | Path) -> MCPConfig:
    """Read and validate MCP server connections from a JSON file.

    Supported shapes:

    1. {"servers": {"name": {"transport": "stdio", ...}}}
    2. {"name": {"transport": "stdio", ...}}
    """
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"MCP config file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid MCP config JSON: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError("MCP config must be a JSON object")

    raw_servers = payload.get("servers")
    if raw_servers is None:
        raw_servers = {
            key: value
            for key, value in payload.items()
            if isinstance(value, dict) and "transport" in value
        }
    if not isinstance(raw_servers, dict):
        raise ValueError("MCP config field 'servers' must be an object")

    connections: dict[str, dict[str, Any]] = {}
    for server_name, raw_connection in raw_servers.items():
        normalized = _normalize_connection(str(server_name), raw_connection)
        if normalized is not None:
            connections[str(server_name)] = normalized

    tool_name_prefix = _coerce_bool(payload.get("tool_name_prefix"), default=True)
    return MCPConfig(connections=connections, tool_name_prefix=tool_name_prefix)


def _normalize_connection(
    server_name: str,
    raw_connection: Any,
) -> dict[str, Any] | None:
    if not SERVER_NAME_PATTERN.fullmatch(server_name):
        raise ValueError(
            f"Invalid MCP server name '{server_name}'. "
            "Use letters, numbers, underscores, or hyphens."
        )
    if not isinstance(raw_connection, dict):
        raise ValueError(f"MCP server '{server_name}' config must be an object")
    if _coerce_bool(raw_connection.get("enabled"), default=True) is False:
        return None

    connection = {
        key: _expand_env_vars(value)
        for key, value in raw_connection.items()
        if key != "enabled"
    }
    transport = str(connection.get("transport") or "").strip()
    transport = "streamable_http" if transport == "streamable-http" else transport
    if transport not in SUPPORTED_TRANSPORTS:
        raise ValueError(
            f"MCP server '{server_name}' has unsupported transport: {transport!r}"
        )
    connection["transport"] = transport

    if transport == "stdio":
        _validate_stdio_connection(server_name, connection)
    else:
        _validate_url_connection(server_name, connection)
        if transport in {"http", "streamable_http"}:
            _coerce_http_timeouts(connection)

    return connection


def _validate_stdio_connection(
    server_name: str,
    connection: dict[str, Any],
) -> None:
    command = str(connection.get("command") or "").strip()
    if not command:
        raise ValueError(f"MCP stdio server '{server_name}' requires 'command'")
    connection["command"] = command

    args = connection.get("args", [])
    if not isinstance(args, list):
        raise ValueError(
            f"MCP stdio server '{server_name}' field 'args' must be a list"
        )
    connection["args"] = [str(item) for item in args]

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
    url = str(connection.get("url") or "").strip()
    if not url:
        raise ValueError(f"MCP server '{server_name}' requires 'url'")
    connection["url"] = url

    headers = connection.get("headers")
    if headers is not None and not isinstance(headers, dict):
        raise ValueError(
            f"MCP server '{server_name}' field 'headers' must be an object"
        )


def _coerce_http_timeouts(connection: dict[str, Any]) -> None:
    for key in ("timeout", "sse_read_timeout"):
        value = connection.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            connection[key] = timedelta(seconds=float(value))


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_VAR_PATTERN.sub(_replace_env_var, value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_env_vars(item) for key, item in value.items()}
    return value


def _replace_env_var(match: re.Match[str]) -> str:
    name = match.group(1)
    default = match.group(2)
    value = os.environ.get(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(f"Environment variable '{name}' is not set")


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    return default
