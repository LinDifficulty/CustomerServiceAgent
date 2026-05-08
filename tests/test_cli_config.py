from __future__ import annotations

import unittest

from rag_server.cli import build_cli_overrides, parse_args


class CLIConfigTests(unittest.TestCase):
    def test_cli_flags_only_override_explicit_values(self) -> None:
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
                "--llm-timeout",
                "0",
                "--memory-top-k",
                "9",
                "--reflection",
                "off",
            ]
        )

        overrides = build_cli_overrides(args)

        self.assertEqual(args.config, "local.toml")
        self.assertEqual(overrides["paths"]["data_dir"], "custom-data")
        self.assertEqual(overrides["paths"]["memory_dir"], "custom-memory")
        self.assertEqual(overrides["agent"]["provider"], "custom-provider")
        self.assertEqual(overrides["agent"]["model"], "agent-model")
        self.assertEqual(overrides["agent"]["model_kwargs"], '{"temperature":0}')
        self.assertEqual(overrides["retrieval"]["embedding_provider"], "emb-provider")
        self.assertEqual(overrides["retrieval"]["embedding_model"], "emb-model")
        self.assertEqual(overrides["retrieval"]["reranker_provider"], "rerank-provider")
        self.assertEqual(overrides["retrieval"]["reranker_model"], "rerank-model")
        self.assertEqual(overrides["retrieval"]["reranker_device"], "cpu")
        self.assertEqual(overrides["llm"]["rewrite_provider"], "rewrite-provider")
        self.assertEqual(overrides["llm"]["rewrite_model"], "rewrite-model")
        self.assertFalse(overrides["retrieval"]["bm25"])
        self.assertTrue(overrides["trace"]["enabled"])
        self.assertFalse(overrides["trace"]["live"])
        self.assertFalse(overrides["cli"]["show_config"])
        self.assertIsNone(overrides["llm"]["timeout_s"])
        self.assertEqual(overrides["memory"]["top_k"], 9)
        self.assertFalse(overrides["agent"]["reflection_enabled"])
        self.assertNotIn("query_rewrite", overrides.get("retrieval", {}))

    def test_live_logs_alias_controls_live_events(self) -> None:
        overrides = build_cli_overrides(parse_args(["--live-logs", "off"]))

        self.assertFalse(overrides["trace"]["live"])

    def test_empty_cli_args_leave_config_unshadowed(self) -> None:
        overrides = build_cli_overrides(parse_args([]))

        self.assertEqual(overrides, {})


if __name__ == "__main__":
    unittest.main()
