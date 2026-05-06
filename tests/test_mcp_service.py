from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import textwrap
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from rag_server.mcp_service import load_mcp_config, load_mcp_tools_from_config


class MCPServiceTest(unittest.TestCase):
    def test_loads_servers_with_env_expansion_and_prefix_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "mcp_servers.json"
            config_path.write_text(
                json.dumps(
                    {
                        "servers": {
                            "crm": {
                                "transport": "http",
                                "url": "http://localhost:8000/mcp",
                                "headers": {
                                    "Authorization": "Bearer ${CRM_MCP_TOKEN}",
                                },
                                "timeout": 10,
                            },
                            "disabled": {
                                "enabled": False,
                                "transport": "stdio",
                                "command": "python",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"CRM_MCP_TOKEN": "secret-token"}):
                config = load_mcp_config(config_path)

            self.assertTrue(config.tool_name_prefix)
            self.assertEqual(sorted(config.connections), ["crm"])
            self.assertEqual(
                config.connections["crm"]["headers"]["Authorization"],
                "Bearer secret-token",
            )
            self.assertEqual(
                config.connections["crm"]["timeout"],
                timedelta(seconds=10),
            )

    def test_supports_top_level_server_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "mcp_servers.json"
            config_path.write_text(
                json.dumps(
                    {
                        "filesystem": {
                            "transport": "stdio",
                            "command": "npx",
                            "args": ["-y", "server"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_mcp_config(config_path)

            self.assertEqual(config.connections["filesystem"]["command"], "npx")
            self.assertEqual(
                config.connections["filesystem"]["args"],
                ["-y", "server"],
            )

    def test_missing_environment_variable_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "mcp_servers.json"
            config_path.write_text(
                json.dumps(
                    {
                        "servers": {
                            "crm": {
                                "transport": "http",
                                "url": "http://localhost:8000/mcp",
                                "headers": {"Authorization": "Bearer ${MISSING_TOKEN}"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "MISSING_TOKEN"):
                load_mcp_config(config_path)

    def test_loads_stdio_tools_with_server_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = Path(temp_dir) / "math_server.py"
            server_path.write_text(
                textwrap.dedent(
                    """
                    from mcp.server.fastmcp import FastMCP

                    mcp = FastMCP("math", log_level="ERROR")

                    @mcp.tool()
                    def add(a: int, b: int) -> int:
                        return a + b

                    if __name__ == "__main__":
                        mcp.run()
                    """
                ),
                encoding="utf-8",
            )
            config_path = Path(temp_dir) / "mcp_servers.json"
            config_path.write_text(
                json.dumps(
                    {
                        "servers": {
                            "math": {
                                "transport": "stdio",
                                "command": sys.executable,
                                "args": [str(server_path)],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = asyncio.run(load_mcp_tools_from_config(config_path))
            tool_names = [tool.name for tool in result.tools]

            self.assertEqual(result.server_names, ["math"])
            self.assertIn("math_add", tool_names)

            add_tool = next(tool for tool in result.tools if tool.name == "math_add")
            output = asyncio.run(add_tool.ainvoke({"a": 2, "b": 3}))
            self.assertIn("5", str(output))


if __name__ == "__main__":
    unittest.main()
