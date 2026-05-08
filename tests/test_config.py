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
                        'provider = "file-provider"',
                        'model = "file-agent"',
                        'user_id = "file-user"',
                        "reflection_enabled = false",
                        "",
                        "[retrieval]",
                        'query_rewrite = "off"',
                        "bm25 = false",
                        'embedding_provider = "file-embedding-provider"',
                        'embedding_model = "file-embedding"',
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
                    "RAG_SERVER_AGENT_PROVIDER": "env-provider",
                    "RAG_SERVER_AGENT_MODEL": "env-agent",
                    "RAG_SERVER_AGENT_MODEL_KWARGS": '{"temperature":0.1}',
                    "RAG_SERVER_EMBEDDING_MODEL": "env-embedding",
                    "RAG_SERVER_REFLECTION": "on",
                    "RAG_SERVER_BM25": "on",
                    "RAG_SERVER_LIVE_EVENTS": "off",
                    "RAG_SERVER_CLI_SHOW_CONFIG": "off",
                },
                overrides={
                    "paths": {"memory_dir": "from_cli_memory"},
                    "agent": {"user_id": "cli-user"},
                },
            )

            self.assertEqual(config.paths.data_dir, "from_env_data")
            self.assertEqual(config.paths.memory_dir, "from_cli_memory")
            self.assertEqual(config.agent.provider, "env-provider")
            self.assertEqual(config.agent.model, "env-agent")
            self.assertEqual(config.agent.model_kwargs["temperature"], 0.1)
            self.assertEqual(config.agent.user_id, "cli-user")
            self.assertTrue(config.agent.reflection_enabled)
            self.assertEqual(config.retrieval.query_rewrite, "off")
            self.assertTrue(config.retrieval.bm25)
            self.assertEqual(config.retrieval.embedding_model, "env-embedding")
            self.assertTrue(config.trace.enabled)
            self.assertFalse(config.trace.live)
            self.assertFalse(config.cli.show_config)

            runtime = config.to_runtime_kwargs()
            self.assertEqual(runtime["data_dir"], "from_env_data")
            self.assertEqual(runtime["memory_dir"], "from_cli_memory")
            self.assertEqual(runtime["agent_provider"], "env-provider")
            self.assertEqual(runtime["agent_model_name"], "env-agent")
            self.assertEqual(runtime["rewrite_provider"], "env-provider")
            self.assertEqual(runtime["rewrite_model_kwargs"]["temperature"], 0.1)
            self.assertEqual(runtime["embedding_model_name"], "env-embedding")
            self.assertTrue(runtime["reflection_enabled"])
            self.assertFalse(runtime["live_events_enabled"])
            self.assertFalse(runtime["show_config"])

    def test_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ConfigError):
            load_app_config(overrides={"agent": {"unknown": "value"}})

    def test_cli_live_log_alias_controls_live_events(self) -> None:
        config = load_app_config(overrides={"cli": {"live_logs": False}})

        self.assertFalse(config.trace.live)
        self.assertFalse(config.to_runtime_kwargs()["live_events_enabled"])

    def test_normalizes_runtime_aliases(self) -> None:
        config = load_app_config(
            overrides={
                "retrieval": {"query_rewrite_mode": "multi_query"},
                "agent": {"reflection": False},
                "llm": {
                    "llm_timeout_s": 0,
                    "query_rewrite_provider": "rewrite-provider",
                    "query_rewrite_model": "rewrite-model",
                },
                "memory": {"memory_top_k": 7},
                "trace": {"live_events": False},
                "cli": {"show_startup_config": False},
            }
        )

        self.assertEqual(config.retrieval.query_rewrite, "multi_query")
        self.assertFalse(config.agent.reflection_enabled)
        self.assertIsNone(config.llm.timeout_s)
        self.assertEqual(config.llm.rewrite_provider, "rewrite-provider")
        self.assertEqual(config.llm.rewrite_model, "rewrite-model")
        self.assertEqual(config.memory.top_k, 7)
        self.assertFalse(config.trace.live)
        self.assertFalse(config.cli.show_config)

    def test_rewrite_model_inherits_agent_kwargs_when_provider_inherits(self) -> None:
        config = load_app_config(
            overrides={
                "agent": {
                    "provider": "custom.chat.Provider",
                    "model_kwargs": {"base_url": "https://example.test/v1"},
                },
                "llm": {"rewrite_model": "small-rewrite-model"},
            }
        )

        runtime = config.to_runtime_kwargs()

        self.assertEqual(runtime["rewrite_provider"], "custom.chat.Provider")
        self.assertEqual(runtime["rewrite_model_name"], "small-rewrite-model")
        self.assertEqual(
            runtime["rewrite_model_kwargs"]["base_url"],
            "https://example.test/v1",
        )


if __name__ == "__main__":
    unittest.main()
