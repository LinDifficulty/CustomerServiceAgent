"""测试 MCP 服务的配置加载和环境变量展开。

包含以下测试场景：
- load_mcp_config 的各种 JSON 格式和验证（servers 键、顶层直写、空配置、禁用的服务器）
- load_mcp_config 的错误处理（无效传输协议、非法服务器名、缺少 command/url、文件不存在、非法 JSON）
- MCP 配置的传输协议和默认前缀
- _expand_env_vars 的环境变量替换、默认值、嵌套结构和非字符串透传
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from rag_server.mcp_service import (
    _expand_env_vars,
    load_mcp_config,
)


class LoadMCPConfigTests(unittest.TestCase):
    """测试从 JSON 文件加载 MCP 配置的各种场景。"""

    def _write_config(self, data: dict) -> str:
        """Write JSON config to a temp file and return its path."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(data, f)
            return f.name

    def test_loads_servers_from_servers_key(self) -> None:
        """从 "servers" 键下加载服务器连接配置。"""
        path = self._write_config({
            "servers": {
                "my-server": {
                    "transport": "stdio",
                    "command": "echo",
                    "args": ["hello"],
                }
            }
        })
        config = load_mcp_config(path)
        os.unlink(path)
        self.assertIn("my-server", config.connections)
        self.assertEqual(config.connections["my-server"]["transport"], "stdio")

    def test_loads_servers_from_top_level(self) -> None:
        """从 JSON 顶层直接解析服务器配置（无需嵌套在 "servers" 键下）。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(
                {
                    "echo-server": {
                        "transport": "stdio",
                        "command": "echo",
                    }
                },
                f,
            )
            f.flush()
            config = load_mcp_config(f.name)

        os.unlink(f.name)
        self.assertIn("echo-server", config.connections)

    def test_empty_config_returns_empty_connections(self) -> None:
        """空配置返回空的 connections 字典。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"servers": {}}, f)
            f.flush()
            config = load_mcp_config(f.name)

        os.unlink(f.name)
        self.assertEqual(config.connections, {})

    def test_disabled_server_is_excluded(self) -> None:
        """enabled: false 的服务器配置应被排除，不加载。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(
                {
                    "servers": {
                        "disabled": {
                            "transport": "stdio",
                            "command": "echo",
                            "enabled": False,
                        }
                    }
                },
                f,
            )
            f.flush()
            config = load_mcp_config(f.name)

        os.unlink(f.name)
        self.assertEqual(config.connections, {})

    def test_invalid_transport_raises(self) -> None:
        """无效的传输协议应抛出 ValueError。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(
                {
                    "servers": {
                        "bad": {"transport": "unknown", "command": "echo"}
                    }
                },
                f,
            )
            f.flush()
            with self.assertRaises(ValueError):
                load_mcp_config(f.name)
            os.unlink(f.name)

    def test_invalid_server_name_raises(self) -> None:
        """非法服务器名称（包含非法字符）应抛出 ValueError。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(
                {
                    "servers": {
                        "bad name!": {"transport": "stdio", "command": "echo"}
                    }
                },
                f,
            )
            f.flush()
            with self.assertRaises(ValueError):
                load_mcp_config(f.name)
            os.unlink(f.name)

    def test_missing_command_raises_for_stdio(self) -> None:
        """stdio 传输协议缺少 command 字段时应抛出 ValueError。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(
                {"servers": {"nocommand": {"transport": "stdio"}}}, f
            )
            f.flush()
            with self.assertRaises(ValueError):
                load_mcp_config(f.name)
            os.unlink(f.name)

    def test_missing_url_raises_for_sse(self) -> None:
        """SSE 传输协议缺少 url 字段时应抛出 ValueError。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"servers": {"nourl": {"transport": "sse"}}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                load_mcp_config(f.name)
            os.unlink(f.name)

    def test_file_not_found_raises(self) -> None:
        """配置文件不存在时应抛出 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            load_mcp_config("/tmp/claude/nonexistent_config.json")

    def test_invalid_json_raises(self) -> None:
        """非法 JSON 内容应抛出 ValueError。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            f.write("{invalid json")
            f.flush()
            with self.assertRaises(ValueError):
                load_mcp_config(f.name)
            os.unlink(f.name)

    def test_tool_name_prefix_default(self) -> None:
        """验证默认启用工具名前缀（tool_name_prefix 为 True）。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"servers": {}}, f)
            f.flush()
            config = load_mcp_config(f.name)

        os.unlink(f.name)
        self.assertTrue(config.tool_name_prefix)

    def test_sse_transport(self) -> None:
        """验证 SSE 传输协议的配置正确加载，包括 transport 和 url 字段。"""
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(
                {
                    "servers": {
                        "sse-srv": {
                            "transport": "sse",
                            "url": "http://localhost:8080",
                        }
                    }
                },
                f,
            )
            f.flush()
            config = load_mcp_config(f.name)

        os.unlink(f.name)
        self.assertEqual(config.connections["sse-srv"]["transport"], "sse")
        self.assertEqual(
            config.connections["sse-srv"]["url"], "http://localhost:8080"
        )


class ExpandEnvVarsTests(unittest.TestCase):
    """测试 MCP 配置中 ${ENV_VAR} 环境变量展开功能。"""

    def test_expand_set_variable(self) -> None:
        """验证已设置的环境变量能正确展开替换。"""
        os.environ["_TEST_MCP_VAR"] = "hello"
        try:
            result = _expand_env_vars("prefix-${_TEST_MCP_VAR}-suffix")
            self.assertEqual(result, "prefix-hello-suffix")
        finally:
            del os.environ["_TEST_MCP_VAR"]

    def test_expand_with_default(self) -> None:
        """验证 ${VAR:-default} 语法：变量未设置时使用默认值。"""
        result = _expand_env_vars("${_MISSING_VAR:-fallback_value}")
        self.assertEqual(result, "fallback_value")

    def test_expand_missing_without_default_raises(self) -> None:
        """验证未设置的必需变量（无默认值）会抛出 ValueError。"""
        with self.assertRaises(ValueError):
            _expand_env_vars("${_UNSET_REQUIRED_VAR}")

    def test_expand_nested_structures(self) -> None:
        """验证递归展开嵌套结构（字典、列表）中的环境变量。"""
        os.environ["_TEST_NESTED"] = "val"
        try:
            result = _expand_env_vars(
                {"key": "${_TEST_NESTED}", "list": ["${_TEST_NESTED}"]}
            )
            self.assertEqual(result, {"key": "val", "list": ["val"]})
        finally:
            del os.environ["_TEST_NESTED"]

    def test_non_string_passthrough(self) -> None:
        """验证非字符串类型（整数、布尔、None）原样返回不处理。"""
        self.assertEqual(_expand_env_vars(42), 42)
        self.assertTrue(_expand_env_vars(True))
        self.assertIsNone(_expand_env_vars(None))


if __name__ == "__main__":
    unittest.main()
