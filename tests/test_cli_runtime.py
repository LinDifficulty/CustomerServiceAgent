from __future__ import annotations

import asyncio
import io
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage

from rag_server.cli import run_cli_async


class FakeStreamingApp:
    async def astream(self, state, stream_mode):
        yield {"agent": {"messages": [AIMessage(content="正常回答")]}}
        yield {"save_memory": None}


class CLIRuntimeTests(unittest.TestCase):
    def test_streaming_ignores_none_node_updates_after_final_answer(self) -> None:
        inputs = iter(["你好", "exit"])

        with (
            patch("rag_server.cli.RAGService", return_value=object()),
            patch(
                "rag_server.cli.build_agent",
                return_value=(FakeStreamingApp(), object(), SystemMessage(content="")),
            ),
            patch("builtins.input", side_effect=lambda prompt="": next(inputs)),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    run_cli_async(
                        query_rewrite_mode="off",
                        memory_enabled=False,
                        skills_enabled=False,
                        mcp_enabled=False,
                        trace_enabled=False,
                        live_events_enabled=False,
                        show_config=False,
                    )
                )
            finally:
                loop.close()
                asyncio.set_event_loop(asyncio.new_event_loop())

        output = stdout.getvalue()
        self.assertIn("正常回答", output)
        self.assertNotIn("大模型调用失败", output)
        self.assertNotIn("NoneType", output)


if __name__ == "__main__":
    unittest.main()
