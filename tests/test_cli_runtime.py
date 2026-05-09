from __future__ import annotations

import asyncio
import io
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from rag_server.cli import (
    CLICompleter,
    CLIInputSession,
    CLIStyle,
    CLIView,
    LLMRetryPolicy,
    PromptToolkitSlashCompleter,
    available_slash_command_specs,
    format_cli_live_event,
    build_retrieval_tool_with_rewrite,
    handle_shortcuts_command,
    handle_unknown_slash_command,
    is_cli_clear_command,
    is_cli_exit_command,
    prompt_toolkit_slash_matches,
    run_cli_async,
)


class FakeStreamingApp:
    async def astream(self, state, stream_mode):
        yield {"agent": {"messages": [AIMessage(content="正常回答")]}}
        yield {"save_memory": None}


class FakeHistoryInspectingApp:
    def __init__(self) -> None:
        self.history_lengths: list[int] = []

    async def astream(self, state, stream_mode):
        self.history_lengths.append(len(state.get("messages", [])))
        yield {"agent": {"messages": [AIMessage(content="回答")]}}
        yield {"save_memory": None}


class FakeToolStreamingApp:
    async def astream(self, state, stream_mode):
        has_prior_tool_result = any(
            isinstance(message, ToolMessage)
            for message in state.get("messages", [])
        )
        if has_prior_tool_result:
            yield {"agent": {"messages": [AIMessage(content="第二轮历史正常")]}}
            yield {"save_memory": None}
            return

        yield {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_product_knowledge",
                                "args": {"question": "第一轮"},
                                "id": "call-1",
                            }
                        ],
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [
                    {
                        "role": "tool",
                        "content": "第一轮工具结果",
                        "tool_call_id": "call-1",
                    }
                ]
            }
        }
        yield {"agent": {"messages": [AIMessage(content="第一轮回答")]}}
        yield {"save_memory": None}


class SyncOnlyRAG:
    def search(self, query: str, **kwargs):
        return [
            {
                "score": 0.9,
                "source": "docs/example.txt",
                "content": f"同步检索结果: {query}",
                "metadata": {"chunk_index": 0},
            }
        ]


class FakeSkill:
    name = "size-guide"
    description = "尺码推荐"
    user_invocable = True


class FakeSkillRegistry:
    errors: list[str] = []

    def list_skills(self):
        return [FakeSkill()]


class FakeReadline:
    __doc__ = "readline"

    def __init__(self) -> None:
        self.completer = None
        self.delims = " \t\n/"
        self.bindings: list[str] = []
        self.buffer = ""
        self.begin = 0

    def get_completer(self):
        return self.completer

    def set_completer(self, completer):
        self.completer = completer

    def get_completer_delims(self):
        return self.delims

    def set_completer_delims(self, delims):
        self.delims = delims

    def parse_and_bind(self, binding):
        self.bindings.append(binding)

    def get_line_buffer(self):
        return self.buffer

    def get_begidx(self):
        return self.begin


class FakeTTY:
    def isatty(self) -> bool:
        return True


class FakeDocument:
    def __init__(self, text_before_cursor: str) -> None:
        self.text_before_cursor = text_before_cursor


class FakePromptToolkitSession:
    def __init__(self) -> None:
        self.calls = []

    async def prompt_async(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return "/memory"


def run_async_case(coro) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


class CLIRuntimeTests(unittest.TestCase):
    def test_exit_command_aliases(self) -> None:
        for value in ("exit", "quit", "/exit", "/quit", "退出", "/退出", " EXIT "):
            self.assertTrue(is_cli_exit_command(value))

        self.assertFalse(is_cli_exit_command("/memory"))

    def test_clear_command_aliases(self) -> None:
        for value in ("clear", "/clear", "清空", "/清空", " CLEAR "):
            self.assertTrue(is_cli_clear_command(value))

        self.assertFalse(is_cli_clear_command("/clear-memory"))

    def test_cli_view_uses_compact_terminal_header_and_prompt(self) -> None:
        view = CLIView(style=CLIStyle(enabled=False), width=52)

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            view.print_startup(
                show_config=False,
                agent_provider="tongyi",
                agent_model_name="qwen-test",
                embedding_provider="dashscope",
                embedding_model_name="embed-test",
                actual_query_rewrite_mode="off",
                rewrite_provider="tongyi",
                rewrite_model_name="qwen-test",
                bm25_enabled=True,
                cross_encoder_enabled=False,
                reranker_provider="cross_encoder",
                reranker_model_name="rerank-test",
                data_dir="data",
                memory_dir="memory",
                memory_enabled=True,
                memory_provider="tongyi",
                memory_model_name="qwen-test",
                retry_policy=LLMRetryPolicy(),
                max_tool_rounds=6,
                max_repeated_tool_calls=2,
                reflection_enabled=True,
                cache_enabled=False,
                cache_connected=False,
                trace_enabled=False,
                trace_path=None,
                live_events_enabled=True,
                skill_registry=None,
                mcp_result=None,
                mcp_tools=[],
                user_id="default_user",
            )

        output = stdout.getvalue()
        self.assertIn("Tulip Agent v", output)
        self.assertIn("tongyi:qwen-test · API Usage", output)
        self.assertIn("skills:0 · mcp:0", output)
        self.assertIn("? for shortcuts", output)
        self.assertIn("-" * 52, output)
        self.assertNotIn("\x1b[", output)
        self.assertEqual(view.input_prompt(), f"\n{'-' * 52}\n> ")

    def test_shortcuts_command_prints_cli_help(self) -> None:
        view = CLIView(style=CLIStyle(enabled=False), width=52)

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            handled = handle_shortcuts_command("?", view, FakeSkillRegistry())

        output = stdout.getvalue()
        self.assertTrue(handled)
        self.assertIn("Shortcuts", output)
        self.assertIn("输入 / 会自动显示可用命令", output)
        self.assertIn("clear / /clear", output)
        self.assertIn("/memory", output)
        self.assertIn("/size-guide", output)
        self.assertIn("exit / quit", output)

    def test_slash_command_specs_include_builtin_and_skills(self) -> None:
        commands = [
            item.command
            for item in available_slash_command_specs(FakeSkillRegistry())
        ]

        self.assertIn("/memory", commands)
        self.assertIn("/clear", commands)
        self.assertIn("/remember-procedure", commands)
        self.assertIn("/size-guide", commands)

    def test_prompt_toolkit_slash_matches_show_all_for_slash(self) -> None:
        matches = prompt_toolkit_slash_matches(
            available_slash_command_specs(FakeSkillRegistry()),
            "/",
        )
        commands = [item.command for item in matches]

        self.assertIn("/memory", commands)
        self.assertIn("/remember", commands)
        self.assertIn("/size-guide", commands)

    def test_prompt_toolkit_slash_matches_filter_prefix_live(self) -> None:
        matches = prompt_toolkit_slash_matches(
            available_slash_command_specs(FakeSkillRegistry()),
            "/mem",
        )

        self.assertEqual([item.command for item in matches], ["/memory"])
        self.assertEqual(
            prompt_toolkit_slash_matches(
                available_slash_command_specs(FakeSkillRegistry()),
                "hello /mem",
            ),
            [],
        )

    def test_prompt_toolkit_completer_returns_live_menu_items(self) -> None:
        completer = PromptToolkitSlashCompleter(
            available_slash_command_specs(FakeSkillRegistry())
        )

        completions = list(completer.get_completions(FakeDocument("/mem"), None))

        self.assertEqual([item.text for item in completions], ["/memory "])
        self.assertEqual(completions[0].start_position, -4)

    def test_prompt_toolkit_completer_supports_async_completion(self) -> None:
        async def collect_completions():
            completer = PromptToolkitSlashCompleter(
                available_slash_command_specs(FakeSkillRegistry())
            )
            return [
                item
                async for item in completer.get_completions_async(
                    FakeDocument("/mem"),
                    None,
                )
            ]

        completions = asyncio.run(collect_completions())

        self.assertEqual([item.text for item in completions], ["/memory "])

    def test_input_session_uses_prompt_toolkit_for_tty_live_completion(self) -> None:
        prompt_session = FakePromptToolkitSession()

        session = CLIInputSession(
            view=CLIView(style=CLIStyle(enabled=False), width=52),
            slash_commands=available_slash_command_specs(FakeSkillRegistry()),
            prompt_toolkit_session_factory=lambda: prompt_session,
            readline_module=None,
            stdin=FakeTTY(),
            stdout=FakeTTY(),
        )

        self.assertEqual(asyncio.run(session.prompt_async()), "/memory")
        self.assertEqual(len(prompt_session.calls), 1)
        self.assertTrue(prompt_session.calls[0][1]["complete_while_typing"])
        self.assertIsInstance(
            prompt_session.calls[0][1]["completer"],
            PromptToolkitSlashCompleter,
        )

    def test_readline_completion_matches_slash_prefix(self) -> None:
        readline = FakeReadline()
        session = CLIInputSession(
            view=CLIView(style=CLIStyle(enabled=False), width=52),
            slash_commands=available_slash_command_specs(FakeSkillRegistry()),
            readline_module=readline,
        )

        self.assertTrue(session.enable_completion())
        readline.buffer = "/rem"
        matches = []
        for state in range(10):
            match = readline.completer("/rem", state)
            if match is None:
                break
            matches.append(match)

        self.assertIn("/remember ", matches)
        self.assertIn("/remember-procedure ", matches)
        self.assertNotIn("/", readline.delims)
        self.assertEqual(readline.bindings, ["tab: complete"])
        session.restore_completion()
        self.assertIn("/", readline.delims)

    def test_readline_completion_ignores_non_initial_words(self) -> None:
        readline = FakeReadline()
        completer = CLICompleter(available_slash_command_specs(FakeSkillRegistry()))
        readline.buffer = "hello /rem"
        readline.begin = 6

        with patch("rag_server.cli._readline", readline):
            self.assertIsNone(completer.complete("/rem", 0))

    def test_unknown_slash_command_prints_suggestions(self) -> None:
        view = CLIView(style=CLIStyle(enabled=False), width=52)

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            handled = handle_unknown_slash_command(
                "/remeber",
                view=view,
                skill_registry=FakeSkillRegistry(),
            )

        output = stdout.getvalue()
        self.assertTrue(handled)
        self.assertIn("Unknown command: /remeber", output)
        self.assertIn("/remember", output)

    def test_known_skill_slash_command_is_not_intercepted(self) -> None:
        self.assertFalse(
            handle_unknown_slash_command(
                "/size-guide 帮我推荐尺码",
                skill_registry=FakeSkillRegistry(),
            )
        )

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
            run_async_case(
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

        output = stdout.getvalue()
        self.assertIn("Tulip Agent", output)
        self.assertIn("Assistant\n正常回答", output)
        self.assertIn("正常回答", output)
        self.assertNotIn("客服:", output)
        self.assertNotIn("大模型调用失败", output)
        self.assertNotIn("NoneType", output)

    def test_streaming_preserves_tool_turn_history_for_followup(self) -> None:
        inputs = iter(["第一轮", "第二轮", "exit"])

        with (
            patch("rag_server.cli.RAGService", return_value=object()),
            patch(
                "rag_server.cli.build_agent",
                return_value=(
                    FakeToolStreamingApp(),
                    object(),
                    SystemMessage(content=""),
                ),
            ),
            patch("builtins.input", side_effect=lambda prompt="": next(inputs)),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            run_async_case(
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

        output = stdout.getvalue()
        self.assertIn("第一轮回答", output)
        self.assertIn("第二轮历史正常", output)
        self.assertNotIn("大模型调用失败", output)

    def test_clear_command_resets_conversation_history(self) -> None:
        inputs = iter(["第一轮", "clear", "第二轮", "exit"])
        app = FakeHistoryInspectingApp()

        with (
            patch("rag_server.cli.RAGService", return_value=object()),
            patch(
                "rag_server.cli.build_agent",
                return_value=(app, object(), SystemMessage(content="")),
            ),
            patch("builtins.input", side_effect=lambda prompt="": next(inputs)),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            run_async_case(
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

        output = stdout.getvalue()
        self.assertEqual(app.history_lengths, [1, 1])
        self.assertIn("Session context cleared.", output)

    def test_live_event_format_uses_tool_blocks(self) -> None:
        output = format_cli_live_event(
            {
                "type": "tool",
                "name": "agent.tool_call_start",
                "payload": {
                    "tool_category": "rag",
                    "tool_name": "search_product_knowledge",
                    "args": {"question": "尺码怎么选"},
                },
            }
        )

        self.assertIn("- RAG search_product_knowledge start", output)
        self.assertIn("args:", output)
        self.assertNotIn("[实时]", output)

    def test_retrieval_tool_falls_back_to_sync_search(self) -> None:
        async def run_case() -> str:
            tool = build_retrieval_tool_with_rewrite(
                SyncOnlyRAG(),
                query_rewrite_mode="off",
            )
            return await tool.ainvoke({"question": "测试问题"})

        output = asyncio.run(run_case())

        self.assertIn("docs/example.txt", output)
        self.assertIn("同步检索结果", output)


if __name__ == "__main__":
    unittest.main()
