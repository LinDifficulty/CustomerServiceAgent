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
    def test_loads_servers_from_servers_key(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(
                {
                    "servers": {
                        "my-server": {
                            "transport": "stdio",
                            "command": "echo",
                            "args": ["hello"],
                        }
                    }
                },
                f,
            )
            f.flush()
            config = load_mcp_config(f.name)

        os.unlink(f.name)
        self.assertIn("my-server", config.connections)
        self.assertEqual(config.connections["my-server"]["transport"], "stdio")

    def test_loads_servers_from_top_level(self) -> None:
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
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"servers": {}}, f)
            f.flush()
            config = load_mcp_config(f.name)

        os.unlink(f.name)
        self.assertEqual(config.connections, {})

    def test_disabled_server_is_excluded(self) -> None:
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
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"servers": {"nourl": {"transport": "sse"}}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                load_mcp_config(f.name)
            os.unlink(f.name)

    def test_file_not_found_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_mcp_config("/tmp/claude/nonexistent_config.json")

    def test_invalid_json_raises(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            f.write("{invalid json")
            f.flush()
            with self.assertRaises(ValueError):
                load_mcp_config(f.name)
            os.unlink(f.name)

    def test_tool_name_prefix_default(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"servers": {}}, f)
            f.flush()
            config = load_mcp_config(f.name)

        os.unlink(f.name)
        self.assertTrue(config.tool_name_prefix)

    def test_sse_transport(self) -> None:
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
    def test_expand_set_variable(self) -> None:
        os.environ["_TEST_MCP_VAR"] = "hello"
        try:
            result = _expand_env_vars("prefix-${_TEST_MCP_VAR}-suffix")
            self.assertEqual(result, "prefix-hello-suffix")
        finally:
            del os.environ["_TEST_MCP_VAR"]

    def test_expand_with_default(self) -> None:
        result = _expand_env_vars("${_MISSING_VAR:-fallback_value}")
        self.assertEqual(result, "fallback_value")

    def test_expand_missing_without_default_raises(self) -> None:
        with self.assertRaises(ValueError):
            _expand_env_vars("${_UNSET_REQUIRED_VAR}")

    def test_expand_nested_structures(self) -> None:
        os.environ["_TEST_NESTED"] = "val"
        try:
            result = _expand_env_vars(
                {"key": "${_TEST_NESTED}", "list": ["${_TEST_NESTED}"]}
            )
            self.assertEqual(result, {"key": "val", "list": ["val"]})
        finally:
            del os.environ["_TEST_NESTED"]

    def test_non_string_passthrough(self) -> None:
        self.assertEqual(_expand_env_vars(42), 42)
        self.assertTrue(_expand_env_vars(True))
        self.assertIsNone(_expand_env_vars(None))


if __name__ == "__main__":
    unittest.main()
