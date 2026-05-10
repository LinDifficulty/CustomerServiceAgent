from __future__ import annotations

import asyncio
import io
import unittest
from unittest.mock import patch

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from rag_server.cli import (
    CLICompleter,
    CLIInputSession,
    CLIStyle,
    CLIThinkingIndicator,
    CLIView,
    LLMRetryPolicy,
    PromptToolkitSlashCompleter,
    available_slash_command_specs,
    build_agent,
    format_cli_live_event,
    CLIStatusEventSink,
    build_retrieval_tool_with_rewrite,
    handle_shortcuts_command,
    handle_unknown_slash_command,
    is_cli_clear_command,
    is_cli_exit_command,
    model_with_streaming_enabled,
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


class FakeSinkThenErrorApp:
    def __init__(self, sink):
        self.sink = sink

    async def astream(self, state, stream_mode):
        result = self.sink("分段回答")
        if hasattr(result, "__await__"):
            await result
        if False:
            yield {}
        raise ValueError(
            'The "function.arguments" parameter of the code model must be in JSON format.'
        )


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


class FakeStreamingModel:
    streaming = False

    def bind_tools(self, tools):
        return self

    async def astream(self, messages, **kwargs):
        yield AIMessageChunk(content="分段")
        yield AIMessageChunk(content="回答")

    async def ainvoke(self, messages):
        return AIMessage(content="分段回答")


class FakeStreamingThenErrorModel(FakeStreamingModel):
    async def astream(self, messages, **kwargs):
        yield AIMessageChunk(content="分段")
        yield AIMessageChunk(content="回答")
        raise ValueError(
            'The "function.arguments" parameter of the code model must be in JSON format.'
        )


class FakeStreamingConnectionDropModel(FakeStreamingModel):
    async def astream(self, messages, **kwargs):
        yield AIMessageChunk(content="半截")
        raise ConnectionError("Remote end closed connection without response")

    async def ainvoke(self, messages):
        return AIMessage(content="普通调用完整回答")


class FakePydanticStyleStreamingModel(FakeStreamingModel):
    def model_copy(self, *, update):
        clone = FakePydanticStyleStreamingModel()
        for key, value in update.items():
            setattr(clone, key, value)
        return clone


class FakeToolThenStreamingModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "search_product_knowledge",
                        "args": '{"question":"第一轮"}',
                        "id": "call-1",
                        "index": 0,
                    }
                ],
            )
            return
        yield AIMessageChunk(content="工具后")
        yield AIMessageChunk(content="回答")

    async def ainvoke(self, messages):
        return AIMessage(content="工具后回答")


class FakePrefaceThenToolErrorModel:
    def bind_tools(self, tools):
        return self

    async def astream(self, messages, **kwargs):
        yield AIMessageChunk(content="我先查一下")
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "search_product_knowledge",
                    "args": '{"question":"羊毛衫尺码"}',
                    "id": "call-1",
                    "index": 0,
                }
            ],
        )
        raise ValueError(
            'The "function.arguments" parameter of the code model must be in JSON format.'
        )

    async def ainvoke(self, messages):
        return AIMessage(content="兜底回答")


class FakePrefaceThenToolSuccessModel:
    """Streaming model: preface text then tool call, then final answer."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(content="让我搜索一下")
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "search_product_knowledge",
                        "args": '{"question":"测试"}',
                        "id": "call-1",
                        "index": 0,
                    }
                ],
            )
            return
        yield AIMessageChunk(content="最终回答")

    async def ainvoke(self, messages):
        return AIMessage(content="最终回答")


class FakeBadSkillToolArgsModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage.model_construct(
                content="",
                tool_calls=[
                    {
                        "name": "read_skill_file",
                        "args": {"name": "sizing-advice"},
                        "id": "bad-read",
                    }
                ],
            )
        has_loaded_skill = any(
            isinstance(message, ToolMessage)
            and "Skill loaded: sizing-advice" in str(message.content)
            for message in messages
        )
        if has_loaded_skill:
            return AIMessage(content="已按尺码 skill 完成回答。")
        has_missing_path_feedback = any(
            isinstance(message, ToolMessage)
            and "read_skill_file 缺少 relative_path" in str(message.content)
            for message in messages
        )
        if has_missing_path_feedback:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_skill",
                        "args": {"name": "sizing-advice"},
                        "id": "load-skill",
                    }
                ],
            )
        return AIMessage(content="已按尺码 skill 完成回答。")


class FakeMissingToolNameModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage.model_construct(
                content="",
                tool_calls=[
                    {
                        "args": {"question": "第一轮"},
                        "id": "missing-name",
                    }
                ],
            )
        has_tool_feedback = any(
            isinstance(message, ToolMessage)
            and "工具调用缺少 name" in str(message.content)
            for message in messages
        )
        if has_tool_feedback:
            return AIMessage(content="已改为直接回答。")
        return AIMessage(content="兜底回答。")


class FakeSkill:
    name = "size-guide"
    description = "尺码推荐"
    user_invocable = True
    allowed_tools: list[str] = []


class FakeSkillRegistry:
    errors: list[str] = []

    def list_skills(self):
        return [FakeSkill()]


class SkillRegistryForBadArgs(FakeSkillRegistry):
    def discovery_prompt(self):
        return (
            "可用 Anthropic-style Skills 如下。\n"
            "- /sizing-advice: 尺码推荐"
        )

    def explicit_invocation_name(self, text):
        return "sizing-advice" if text.strip().startswith("/sizing-advice") else None

    def render_explicit_skill_context(self, text):
        return ""

    def get_skill(self, name):
        return FakeSkill() if name == "sizing-advice" else None

    def load_skill(self, name):
        return "Skill loaded: sizing-advice\n\nSKILL.md instructions:\n先询问身高体重。"

    def list_supporting_files(self, name):
        return ["checklist.md"]

    def read_supporting_file(self, name, relative_path):
        return f"Skill file: {name}/{relative_path}"


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
                stream_output_enabled=True,
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

    def test_agent_streams_final_answer_chunks_to_sink(self) -> None:
        async def run_case() -> tuple[str, str]:
            deltas: list[str] = []
            app, _, _ = build_agent(
                SyncOnlyRAG(),
                query_rewrite_mode="off",
                skills_enabled=False,
                memory_service=None,
                memory_extractor=None,
                agent_model=FakeStreamingModel(),
                reflection_enabled=True,
                stream_output_enabled=True,
                output_delta_sink=deltas.append,
            )
            result = await app.ainvoke(
                {
                    "messages": [HumanMessage(content="你好")],
                    "user_id": "user",
                }
            )
            return "".join(deltas), result["messages"][-1].content

        streamed, final = asyncio.run(run_case())

        self.assertEqual(streamed, "分段回答")
        self.assertEqual(final, "分段回答")

    def test_model_with_streaming_enabled_uses_model_copy(self) -> None:
        model = FakePydanticStyleStreamingModel()

        streamed_model = model_with_streaming_enabled(model)

        self.assertIsNot(streamed_model, model)
        self.assertFalse(model.streaming)
        self.assertTrue(streamed_model.streaming)

    def test_agent_streaming_preserves_tool_calls_before_final_answer(self) -> None:
        async def run_case() -> tuple[str, str]:
            deltas: list[str] = []
            app, _, _ = build_agent(
                SyncOnlyRAG(),
                query_rewrite_mode="off",
                skills_enabled=False,
                memory_service=None,
                memory_extractor=None,
                agent_model=FakeToolThenStreamingModel(),
                reflection_enabled=False,
                stream_output_enabled=True,
                output_delta_sink=deltas.append,
            )
            result = await app.ainvoke(
                {
                    "messages": [HumanMessage(content="第一轮")],
                    "user_id": "user",
                }
            )
            return "".join(deltas), result["messages"][-1].content

        streamed, final = asyncio.run(run_case())

        self.assertEqual(streamed, "工具后回答")
        self.assertEqual(final, "工具后回答")

    def test_agent_streams_preface_before_tool_call_and_keeps_it(self) -> None:
        async def run_case() -> list[str]:
            deltas: list[str] = []
            app, _, _ = build_agent(
                SyncOnlyRAG(),
                query_rewrite_mode="off",
                skills_enabled=False,
                memory_service=None,
                memory_extractor=None,
                agent_model=FakePrefaceThenToolErrorModel(),
                reflection_enabled=False,
                stream_output_enabled=True,
                output_delta_sink=deltas.append,
            )
            with self.assertRaises(ValueError):
                await app.ainvoke(
                    {
                        "messages": [HumanMessage(content="羊毛衫尺码")],
                        "user_id": "user",
                    }
                )
            return deltas

        deltas = asyncio.run(run_case())

        # 前置引导文本在 tool call 出现前已发送到 sink 并保留显示。
        self.assertEqual("".join(deltas), "我先查一下")

    def test_agent_streams_preface_and_final_answer_through_tool_loop(self) -> None:
        async def run_case() -> tuple[str, str]:
            deltas: list[str] = []
            app, _, _ = build_agent(
                SyncOnlyRAG(),
                query_rewrite_mode="off",
                skills_enabled=False,
                memory_service=None,
                memory_extractor=None,
                agent_model=FakePrefaceThenToolSuccessModel(),
                reflection_enabled=False,
                stream_output_enabled=True,
                output_delta_sink=deltas.append,
            )
            result = await app.ainvoke(
                {
                    "messages": [HumanMessage(content="测试")],
                    "user_id": "user",
                }
            )
            return "".join(deltas), result["messages"][-1].content

        streamed, final = asyncio.run(run_case())

        # 前置引导文本和最终回答都应保留在 stream 中
        self.assertEqual(streamed, "让我搜索一下最终回答")
        self.assertEqual(final, "最终回答")

    def test_missing_skill_file_path_feedback_is_returned_to_model_for_repair(self) -> None:
        async def run_case() -> tuple[str, list[ToolMessage]]:
            app, _, _ = build_agent(
                SyncOnlyRAG(),
                query_rewrite_mode="off",
                skills_enabled=True,
                skill_registry=SkillRegistryForBadArgs(),
                memory_service=None,
                memory_extractor=None,
                agent_model=FakeBadSkillToolArgsModel(),
                reflection_enabled=False,
                stream_output_enabled=False,
            )
            result = await app.ainvoke(
                {
                    "messages": [HumanMessage(content="/sizing-advice 帮我推荐尺码")],
                    "user_id": "user",
                }
            )
            tool_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
            ]
            return result["messages"][-1].content, tool_messages

        final, tool_messages = asyncio.run(run_case())

        self.assertEqual(final, "已按尺码 skill 完成回答。")
        self.assertTrue(
            any("read_skill_file 缺少 relative_path" in str(item.content) for item in tool_messages)
        )
        self.assertTrue(any("Skill loaded: sizing-advice" in str(item.content) for item in tool_messages))

    def test_missing_tool_call_name_is_returned_to_model_for_repair(self) -> None:
        async def run_case() -> tuple[str, list[ToolMessage]]:
            app, _, _ = build_agent(
                SyncOnlyRAG(),
                query_rewrite_mode="off",
                skills_enabled=False,
                memory_service=None,
                memory_extractor=None,
                agent_model=FakeMissingToolNameModel(),
                reflection_enabled=False,
                stream_output_enabled=False,
            )
            result = await app.ainvoke(
                {
                    "messages": [HumanMessage(content="第一轮")],
                    "user_id": "user",
                }
            )
            tool_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
            ]
            return result["messages"][-1].content, tool_messages

        final, tool_messages = asyncio.run(run_case())

        self.assertEqual(final, "已改为直接回答。")
        self.assertTrue(any("工具调用缺少 name" in str(item.content) for item in tool_messages))

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

    def test_thinking_indicator_animates_and_clears_on_tty(self) -> None:
        async def run_case() -> str:
            stream = io.StringIO()
            stream.isatty = lambda: True
            view = CLIView(style=CLIStyle(enabled=True), width=52)
            indicator = CLIThinkingIndicator(
                view,
                text="正在分析问题...",
                interval_s=0.001,
            )
            with patch("sys.stdout", stream):
                indicator.start()
                indicator.update("找到相关知识...")
                await asyncio.sleep(0.012)
                await indicator.stop()
            return stream.getvalue()

        output = asyncio.run(run_case())

        self.assertIn("正在分析问题", output)
        self.assertIn("找到相关知识", output)
        self.assertIn(CLIStyle.ACCENT, output)
        self.assertIn(CLIStyle.LOGO_GREEN, output)
        self.assertIn("\r\033[K", output)

    def test_status_event_sink_maps_real_retrieval_events(self) -> None:
        self.assertEqual(
            CLIStatusEventSink.status_for_record(
                {
                    "type": "tool",
                    "name": "agent.tool_call_start",
                    "payload": {"tool_category": "rag"},
                }
            ),
            "正在检索相关知识...",
        )
        self.assertEqual(
            CLIStatusEventSink.status_for_record(
                {
                    "type": "tool",
                    "name": "tool.search_product_knowledge",
                    "payload": {"result_count": 2},
                }
            ),
            "找到相关知识...",
        )
        self.assertEqual(
            CLIStatusEventSink.status_for_record(
                {
                    "type": "tool",
                    "name": "tool.search_product_knowledge",
                    "payload": {"result_count": 0},
                }
            ),
            "未找到相关知识，继续直接回答...",
        )

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
        self.assertIn("正在分析问题", output)
        self.assertIn("正在组织答案", output)
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

    def test_cli_runtime_uses_model_token_stream_without_duplicate_final(self) -> None:
        inputs = iter(["你好", "exit"])

        with (
            patch("rag_server.cli.RAGService", return_value=SyncOnlyRAG()),
            patch("rag_server.cli.create_chat_model", return_value=FakeStreamingModel()),
            patch("builtins.input", side_effect=lambda prompt="": next(inputs)),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            run_async_case(
                run_cli_async(
                    query_rewrite_mode="off",
                    memory_enabled=False,
                    skills_enabled=False,
                    mcp_enabled=False,
                    reflection_enabled=False,
                    trace_enabled=False,
                    live_events_enabled=False,
                    show_config=False,
                    stream_output_enabled=True,
                )
            )

        output = stdout.getvalue()
        self.assertIn("Assistant\n分段回答", output)
        self.assertEqual(output.count("分段回答"), 1)
        self.assertNotIn("大模型调用失败", output)

    def test_cli_keeps_streamed_answer_when_provider_errors_after_tokens(self) -> None:
        inputs = iter(["你好", "exit"])

        with (
            patch("rag_server.cli.RAGService", return_value=SyncOnlyRAG()),
            patch(
                "rag_server.cli.create_chat_model",
                return_value=FakeStreamingThenErrorModel(),
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
                    reflection_enabled=False,
                    trace_enabled=False,
                    live_events_enabled=False,
                    show_config=False,
                    stream_output_enabled=True,
                )
            )

        output = stdout.getvalue()
        self.assertIn("Assistant\n分段回答", output)
        self.assertNotIn("Error", output)
        self.assertNotIn("大模型调用失败", output)
        self.assertNotIn("function.arguments", output)

    def test_cli_falls_back_to_invoke_when_stream_connection_drops(self) -> None:
        inputs = iter(["你好", "exit"])

        with (
            patch("rag_server.cli.RAGService", return_value=SyncOnlyRAG()),
            patch(
                "rag_server.cli.create_chat_model",
                return_value=FakeStreamingConnectionDropModel(),
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
                    reflection_enabled=False,
                    trace_enabled=False,
                    live_events_enabled=False,
                    show_config=False,
                    stream_output_enabled=True,
                )
            )

        output = stdout.getvalue()
        self.assertIn("Assistant\n", output)
        self.assertIn("半截", output)
        self.assertIn("\r\x1b[K", output)
        self.assertIn("普通调用完整回答", output)
        self.assertNotIn("ConnectionError", output)
        self.assertNotIn("大模型调用失败", output)

    def test_cli_reports_graph_error_when_sink_output_has_no_final_message(self) -> None:
        inputs = iter(["你好", "exit"])
        captured_sink = {}

        def fake_build_agent(*args, **kwargs):
            captured_sink["sink"] = kwargs["output_delta_sink"]
            return (
                FakeSinkThenErrorApp(captured_sink["sink"]),
                object(),
                SystemMessage(content=""),
            )

        with (
            patch("rag_server.cli.RAGService", return_value=object()),
            patch("rag_server.cli.build_agent", side_effect=fake_build_agent),
            patch("builtins.input", side_effect=lambda prompt="": next(inputs)),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            run_async_case(
                run_cli_async(
                    query_rewrite_mode="off",
                    memory_enabled=False,
                    skills_enabled=False,
                    mcp_enabled=False,
                    reflection_enabled=False,
                    trace_enabled=False,
                    live_events_enabled=False,
                    show_config=False,
                    stream_output_enabled=True,
                )
            )

        output = stdout.getvalue()
        self.assertIn("Assistant\n分段回答", output)
        self.assertIn("Error", output)
        self.assertIn("大模型调用失败", output)
        self.assertIn("function.arguments", output)

    def test_cli_status_reflects_real_rag_retrieval_before_answer(self) -> None:
        inputs = iter(["第一轮", "exit"])

        with (
            patch("rag_server.cli.RAGService", return_value=SyncOnlyRAG()),
            patch(
                "rag_server.cli.create_chat_model",
                return_value=FakeToolThenStreamingModel(),
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
                    reflection_enabled=False,
                    trace_enabled=False,
                    live_events_enabled=False,
                    show_config=False,
                    stream_output_enabled=True,
                )
            )

        output = stdout.getvalue()
        self.assertIn("正在分析问题", output)
        self.assertIn("正在检索相关知识", output)
        self.assertIn("找到相关知识", output)
        self.assertIn("正在搜索知识库", output)
        self.assertIn("Assistant\n工具后回答", output)
        self.assertNotIn("- RAG", output)
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
