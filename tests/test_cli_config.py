"""测试 CLI 命令行参数解析与配置覆盖逻辑，以及终端启动清屏行为。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from rag_server.cli import build_cli_overrides, clear_terminal_startup, main, parse_args


class CLIConfigTests(unittest.TestCase):
    """CLI 配置层测试：验证命令行参数到配置覆盖的映射、别名处理、终端启动清屏逻辑。"""

    def test_cli_flags_only_override_explicit_values(self) -> None:
        """显式传入的所有命令行参数应正确映射为配置覆盖项，未传入的参数不应出现在覆写结果中。"""
        args = parse_args(
            [
                "--config",
                "local.toml",
                "--data-dir",
                "custom-data",
                "--memory-dir",
                "custom-memory",
                "--agent-provider",
                "custom-provider",
                "--agent-model",
                "agent-model",
                "--agent-model-kwargs",
                '{"temperature":0}',
                "--embedding-provider",
                "emb-provider",
                "--embedding-model",
                "emb-model",
                "--reranker-provider",
                "rerank-provider",
                "--reranker-model",
                "rerank-model",
                "--reranker-device",
                "cpu",
                "--rewrite-provider",
                "rewrite-provider",
                "--rewrite-model",
                "rewrite-model",
                "--bm25",
                "off",
                "--trace",
                "on",
                "--live-events",
                "off",
                "--show-config",
                "off",
                "--stream-output",
                "off",
                "--llm-timeout",
                "0",
                "--memory-top-k",
                "9",
                "--reflection",
                "off",
                "--cache",
                "on",
                "--redis-url",
                "redis://localhost:6380/1",
                "--cache-namespace",
                "cli-test",
                "--cache-retrieval-ttl",
                "120",
            ]
        )

        overrides = build_cli_overrides(args)

        self.assertEqual(args.config, "local.toml")
        # 路径相关的覆盖
        self.assertEqual(overrides["paths"]["data_dir"], "custom-data")
        self.assertEqual(overrides["paths"]["memory_dir"], "custom-memory")
        # Agent 相关的覆盖
        self.assertEqual(overrides["agent"]["provider"], "custom-provider")
        self.assertEqual(overrides["agent"]["model"], "agent-model")
        self.assertEqual(overrides["agent"]["model_kwargs"], '{"temperature":0}')
        # 检索相关的覆盖
        self.assertEqual(overrides["retrieval"]["embedding_provider"], "emb-provider")
        self.assertEqual(overrides["retrieval"]["embedding_model"], "emb-model")
        self.assertEqual(overrides["retrieval"]["reranker_provider"], "rerank-provider")
        self.assertEqual(overrides["retrieval"]["reranker_model"], "rerank-model")
        self.assertEqual(overrides["retrieval"]["reranker_device"], "cpu")
        # LLM 相关的覆盖
        self.assertEqual(overrides["llm"]["rewrite_provider"], "rewrite-provider")
        self.assertEqual(overrides["llm"]["rewrite_model"], "rewrite-model")
        self.assertFalse(overrides["retrieval"]["bm25"])
        self.assertTrue(overrides["trace"]["enabled"])
        self.assertFalse(overrides["trace"]["live"])
        self.assertFalse(overrides["cli"]["show_config"])
        self.assertFalse(overrides["cli"]["stream_output"])
        self.assertIsNone(overrides["llm"]["timeout_s"])
        # LLM timeout 为 0 时应转换为 None
        self.assertEqual(overrides["memory"]["top_k"], 9)
        self.assertFalse(overrides["agent"]["reflection_enabled"])
        # 缓存相关的覆盖
        self.assertTrue(overrides["cache"]["enabled"])
        self.assertEqual(overrides["cache"]["redis_url"], "redis://localhost:6380/1")
        self.assertEqual(overrides["cache"]["namespace"], "cli-test")
        self.assertEqual(overrides["cache"]["retrieval_ttl_s"], 120)
        # query_rewrite 未传入，不应出现在 retrieval 的覆写结果中
        self.assertNotIn("query_rewrite", overrides.get("retrieval", {}))

    def test_live_logs_alias_controls_live_events(self) -> None:
        """--live-logs 别名参数应正确控制 trace.live 配置项。"""
        overrides = build_cli_overrides(parse_args(["--live-logs", "off"]))

        self.assertFalse(overrides["trace"]["live"])

    def test_empty_cli_args_leave_config_unshadowed(self) -> None:
        """无命令行参数时覆写结果为空字典，不影响配置文件默认值。"""
        overrides = build_cli_overrides(parse_args([]))

        self.assertEqual(overrides, {})

    def test_clear_terminal_startup_uses_system_clear_for_tty(self) -> None:
        """在 TTY 终端环境下启动时应调用系统 clear 命令清屏。"""
        stream = Mock()
        stream.isatty.return_value = True

        with patch("rag_server.cli.os.system") as system:
            clear_terminal_startup(stream)

        system.assert_called_once_with("clear")

    def test_clear_terminal_startup_skips_non_tty(self) -> None:
        """非 TTY 环境下（如管道重定向）启动时不执行清屏操作。"""
        stream = Mock()
        stream.isatty.return_value = False

        with patch("rag_server.cli.os.system") as system:
            clear_terminal_startup(stream)

        system.assert_not_called()

    def test_main_clears_terminal_after_config_load_before_cli_run(self) -> None:
        """main() 的执行顺序应为：加载配置 → 清屏 → 运行 CLI，清屏在配置加载之后、CLI 运行之前。"""
        calls: list[str] = []
        config = Mock()
        config.to_runtime_kwargs.return_value = {"query_rewrite_mode": "off"}

        def fake_load_app_config(*args, **kwargs):
            calls.append("load_config")
            return config

        def fake_clear_terminal_startup():
            calls.append("clear")

        def fake_run_cli(**kwargs):
            calls.append("run_cli")

        with (
            patch("rag_server.cli.load_app_config", side_effect=fake_load_app_config),
            patch(
                "rag_server.cli.clear_terminal_startup",
                side_effect=fake_clear_terminal_startup,
            ),
            patch("rag_server.cli.run_cli", side_effect=fake_run_cli),
        ):
            main([])

        self.assertEqual(calls, ["load_config", "clear", "run_cli"])
        config.to_runtime_kwargs.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
