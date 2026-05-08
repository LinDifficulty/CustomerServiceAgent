from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.config import ConfigError, load_app_config


class AppConfigTests(unittest.TestCase):
    def test_loads_defaults_file_env_and_overrides_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "rag.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'data_dir = "from_file_data"',
                        'memory_dir = "from_file_memory"',
                        "",
                        "[agent]",
                        'model = "file-agent"',
                        'user_id = "file-user"',
                        "reflection_enabled = false",
                        "",
                        "[retrieval]",
                        'query_rewrite = "off"',
                        "bm25 = false",
                        "",
                        "[trace]",
                        "enabled = true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_app_config(
                config_path,
                env={
                    "RAG_SERVER_DATA_DIR": "from_env_data",
                    "RAG_SERVER_AGENT_MODEL": "env-agent",
                    "RAG_SERVER_REFLECTION": "on",
                    "RAG_SERVER_BM25": "on",
                    "RAG_SERVER_LIVE_EVENTS": "off",
                },
                overrides={
                    "paths": {"memory_dir": "from_cli_memory"},
                    "agent": {"user_id": "cli-user"},
                },
            )

            self.assertEqual(config.paths.data_dir, "from_env_data")
            self.assertEqual(config.paths.memory_dir, "from_cli_memory")
            self.assertEqual(config.agent.model, "env-agent")
            self.assertEqual(config.agent.user_id, "cli-user")
            self.assertTrue(config.agent.reflection_enabled)
            self.assertEqual(config.retrieval.query_rewrite, "off")
            self.assertTrue(config.retrieval.bm25)
            self.assertTrue(config.trace.enabled)
            self.assertFalse(config.trace.live)

            runtime = config.to_runtime_kwargs()
            self.assertEqual(runtime["data_dir"], "from_env_data")
            self.assertEqual(runtime["memory_dir"], "from_cli_memory")
            self.assertEqual(runtime["agent_model_name"], "env-agent")
            self.assertTrue(runtime["reflection_enabled"])
            self.assertFalse(runtime["live_events_enabled"])

    def test_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ConfigError):
            load_app_config(overrides={"agent": {"unknown": "value"}})

    def test_normalizes_runtime_aliases(self) -> None:
        config = load_app_config(
            overrides={
                "retrieval": {"query_rewrite_mode": "multi_query"},
                "agent": {"reflection": False},
                "llm": {"llm_timeout_s": 0},
                "memory": {"memory_top_k": 7},
                "trace": {"live_events": False},
            }
        )

        self.assertEqual(config.retrieval.query_rewrite, "multi_query")
        self.assertFalse(config.agent.reflection_enabled)
        self.assertIsNone(config.llm.timeout_s)
        self.assertEqual(config.memory.top_k, 7)
        self.assertFalse(config.trace.live)


if __name__ == "__main__":
    unittest.main()
