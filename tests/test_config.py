"""测试应用配置加载、合并和校验逻辑，包括 TOML 文件、环境变量、CLI 覆写的优先级链和别名规范化。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.config import ConfigError, load_app_config


class AppConfigTests(unittest.TestCase):
    """应用配置单元测试：验证默认值、TOML 文件 → 环境变量 → CLI 覆写的三级优先级覆盖，以及配置别名规范化和非法键校验。"""

    def test_cli_noise_defaults_are_off(self) -> None:
        """默认配置下，终端噪音开关（live_events、show_config）应关闭，流式输出应开启。"""
        config = load_app_config()

        self.assertFalse(config.trace.live)
        self.assertFalse(config.cli.show_config)
        self.assertTrue(config.cli.stream_output)
        self.assertFalse(config.to_runtime_kwargs()["live_events_enabled"])
        self.assertFalse(config.to_runtime_kwargs()["show_config"])
        self.assertTrue(config.to_runtime_kwargs()["stream_output_enabled"])

    def test_loads_defaults_file_env_and_overrides_in_order(self) -> None:
        """验证配置三级优先级：CLI overrides > 环境变量 > TOML 文件 > 默认值，所有字段均按此顺序被最终值覆盖。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "rag.toml"
            # 写入 TOML 文件，作为第一级配置来源
            config_path.write_text(
                """\
[paths]
data_dir = "from_file_data"
memory_dir = "from_file_memory"

[agent]
provider = "file-provider"
model = "file-agent"
user_id = "file-user"
reflection_enabled = false

[retrieval]
query_rewrite = "off"
bm25 = false
embedding_provider = "file-embedding-provider"
embedding_model = "file-embedding"

[trace]
enabled = true
""",
                encoding="utf-8",
            )

            config = load_app_config(
                config_path,
                # 环境变量作为第二级配置来源
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
                    "RAG_SERVER_STREAM_OUTPUT": "off",
                },
                # CLI overrides 作为最高优先级
                overrides={
                    "paths": {"memory_dir": "from_cli_memory"},
                    "agent": {"user_id": "cli-user"},
                },
            )

            # 验证 paths：data_dir 被环境变量覆盖，memory_dir 被 CLI 覆盖
            self.assertEqual(config.paths.data_dir, "from_env_data")
            self.assertEqual(config.paths.memory_dir, "from_cli_memory")
            # 验证 agent：provider/model 被环境变量覆盖，user_id 被 CLI 覆盖
            self.assertEqual(config.agent.provider, "env-provider")
            self.assertEqual(config.agent.model, "env-agent")
            self.assertEqual(config.agent.model_kwargs["temperature"], 0.1)
            self.assertEqual(config.agent.user_id, "cli-user")
            self.assertTrue(config.agent.reflection_enabled)
            # 验证 retrieval：query_rewrite 来自文件，bm25 被环境变量反转
            self.assertEqual(config.retrieval.query_rewrite, "off")
            self.assertTrue(config.retrieval.bm25)
            self.assertEqual(config.retrieval.embedding_model, "env-embedding")
            self.assertTrue(config.trace.enabled)
            self.assertFalse(config.trace.live)
            self.assertFalse(config.cli.show_config)
            self.assertFalse(config.cli.stream_output)
            self.assertTrue(config.cache.enabled)

            # 验证 to_runtime_kwargs 转换的正确性
            runtime = config.to_runtime_kwargs()
            self.assertEqual(runtime["data_dir"], "from_env_data")
            self.assertEqual(runtime["memory_dir"], "from_cli_memory")
            self.assertEqual(runtime["agent_provider"], "env-provider")
            self.assertEqual(runtime["agent_model_name"], "env-agent")
            # rewrite 默认继承 agent 的 provider 和 kwargs
            self.assertEqual(runtime["rewrite_provider"], "env-provider")
            self.assertEqual(runtime["rewrite_model_kwargs"]["temperature"], 0.1)
            self.assertEqual(runtime["embedding_model_name"], "env-embedding")
            self.assertTrue(runtime["reflection_enabled"])
            self.assertFalse(runtime["live_events_enabled"])
            self.assertFalse(runtime["show_config"])
            self.assertFalse(runtime["stream_output_enabled"])
            self.assertTrue(runtime["cache_enabled"])

    def test_rejects_unknown_keys(self) -> None:
        """传入未知的配置键时应抛出 ConfigError 异常。"""
        with self.assertRaises(ConfigError):
            load_app_config(overrides={"agent": {"unknown": "value"}})

    def test_cli_live_log_alias_controls_live_events(self) -> None:
        """cli.live_logs 别名应正确映射并控制 trace.live 配置项。"""
        config = load_app_config(overrides={"cli": {"live_logs": False}})

        self.assertFalse(config.trace.live)
        self.assertFalse(config.to_runtime_kwargs()["live_events_enabled"])

    def test_normalizes_runtime_aliases(self) -> None:
        """验证所有配置别名（如 query_rewrite_mode、reflection、llm_timeout_s 等）均被正确规范化到标准键名。"""
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
                "cli": {"show_startup_config": False, "streaming": False},
            }
        )

        # query_rewrite_mode 别名应映射到 query_rewrite
        self.assertEqual(config.retrieval.query_rewrite, "multi_query")
        # reflection 别名应映射到 reflection_enabled
        self.assertFalse(config.agent.reflection_enabled)
        # llm_timeout_s 为 0 时应被规范化为 None（无超时）
        self.assertIsNone(config.llm.timeout_s)
        self.assertEqual(config.llm.rewrite_provider, "rewrite-provider")
        self.assertEqual(config.llm.rewrite_model, "rewrite-model")
        self.assertEqual(config.memory.top_k, 7)
        self.assertFalse(config.trace.live)
        # show_startup_config/streaming 别名应映射到 show_config/stream_output
        self.assertFalse(config.cli.show_config)
        self.assertFalse(config.cli.stream_output)

    def test_rewrite_model_inherits_agent_kwargs_when_provider_inherits(self) -> None:
        """当 rewrite 未单独指定 provider 时，应从 agent 继承 provider 和 model_kwargs（如 base_url）。"""
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

    def test_cache_config_from_env_and_overrides(self) -> None:
        """验证缓存配置从环境变量和 overrides 正确合并：env 提供基本值，overrides 补充 embedding_ttl 和 memory_ttl。"""
        config = load_app_config(
            env={
                "RAG_SERVER_CACHE": "on",
                "RAG_SERVER_REDIS_URL": "redis://localhost:6380/2",
                "RAG_SERVER_CACHE_NAMESPACE": "test-rag",
                "RAG_SERVER_CACHE_RETRIEVAL_TTL": "120",
            },
            overrides={
                "cache": {
                    "embedding_ttl_s": 30,
                    "memory_ttl_s": 5,
                }
            },
        )

        runtime = config.to_runtime_kwargs()

        self.assertTrue(config.cache.enabled)
        self.assertEqual(config.cache.redis_url, "redis://localhost:6380/2")
        self.assertEqual(config.cache.namespace, "test-rag")
        self.assertEqual(config.cache.retrieval_ttl_s, 120)
        self.assertEqual(config.cache.embedding_ttl_s, 30)
        self.assertEqual(config.cache.memory_ttl_s, 5)
        self.assertTrue(runtime["cache_enabled"])
        self.assertEqual(runtime["cache_namespace"], "test-rag")


if __name__ == "__main__":
    unittest.main()
