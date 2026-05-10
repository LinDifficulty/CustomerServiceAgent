"""测试 CLI 运行时行为，包括流式输出、工具调用、斜杠命令补全、思考状态动画、会话清空等交互场景。"""

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
    """模拟的流式 Agent 应用，直接返回 AIMessage 最终回答。"""

    async def astream(self, state, stream_mode):
        yield {"agent": {"messages": [AIMessage(content="正常回答")]}}
        yield {"save_memory": None}


class FakeHistoryInspectingApp:
    """模拟的 Agent 应用，记录每次调用时的历史消息长度，用于验证 clear 命令是否清空了上下文。"""
    def __init__(self) -> None:
        self.history_lengths: list[int] = []

    async def astream(self, state, stream_mode):
        self.history_lengths.append(len(state.get("messages", [])))
        yield {"agent": {"messages": [AIMessage(content="回答")]}}
        yield {"save_memory": None}


class FakeToolStreamingApp:
    """模拟先工具调用再生成最终回答的流式 Agent，验证多轮工具调用历史被正确保留。"""

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
    """模拟先将流式内容写入 sink 后抛出异常的 Agent，用于测试异常时保留已流式输出的内容。"""

    def __init__(self, sink):
        self.sink = sink

    async def astream(self, state, stream_mode):
        result = self.sink("分段回答")
        if hasattr(result, "__await__"):
            await result
        # This `yield` is never reached but makes Python treat astream as an
        # async generator, which `async for` requires.
        if False:  # pragma: no cover
            yield {}
        raise ValueError(
            'The "function.arguments" parameter of the code model must be in JSON format.'
        )


class SyncOnlyRAG:
    """仅支持同步检索的假 RAG 服务，用于测试 retrieval tool 的同步回退。"""

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
    """模拟流式模型，逐 chunk 返回内容，用于测试 token-level 流式输出。"""

    streaming = False

    def bind_tools(self, tools):
        return self

    async def astream(self, messages, **kwargs):
        yield AIMessageChunk(content="分段")
        yield AIMessageChunk(content="回答")

    async def ainvoke(self, messages):
        return AIMessage(content="分段回答")


class FakeStreamingThenErrorModel(FakeStreamingModel):
    """模拟流式输出完毕后抛出异常的模型，验证已输出的内容是否被保留。"""
    async def astream(self, messages, **kwargs):
        yield AIMessageChunk(content="分段")
        yield AIMessageChunk(content="回答")
        raise ValueError(
            'The "function.arguments" parameter of the code model must be in JSON format.'
        )


class FakeStreamingConnectionDropModel(FakeStreamingModel):
    """模拟流式传输中途连接断开的模型，验证断开后自动回退到普通 ainvoke 获取完整回答。"""

    async def astream(self, messages, **kwargs):
        yield AIMessageChunk(content="半截")
        raise ConnectionError("Remote end closed connection without response")

    async def ainvoke(self, messages):
        return AIMessage(content="普通调用完整回答")


class FakePydanticStyleStreamingModel(FakeStreamingModel):
    """支持 model_copy 的假流式模型，用于测试 model_with_streaming_enabled 使用 model_copy 创建副本。"""
    def model_copy(self, *, update):
        clone = FakePydanticStyleStreamingModel()
        for key, value in update.items():
            setattr(clone, key, value)
        return clone


class FakeToolThenStreamingModel:
    """模拟先流式返回工具调用 chunk、第二轮再流式返回最终回答的模型，用于测试工具调用后再生成的流式场景。"""

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
    """模拟先输出前置引导文本、返回 tool_call chunk 后抛出异常的模型，验证异常时前置文本是否被保留。"""

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
    """模拟先输出前置引导文本、工具调用再返回最终回答的正常流程，验证完整流式链路。"""

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
    """模拟 skill 工具调用参数缺失（缺 relative_path）时 Agent 的自我修复流程：收到错误反馈后修正调用。"""

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
    """模拟工具调用缺少 name 字段时的修复流程：收到反馈后改为直接回答而不使用工具。"""

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
    """模拟的 Skill 定义对象，用于 skill 补全和注册测试。"""

    name = "size-guide"
    description = "尺码推荐"
    user_invocable = True
    allowed_tools: list[str] = []


class FakeSkillRegistry:
    """模拟的 Skill 注册表，提供一个可被发现的伪 skill。"""

    errors: list[str] = []

    def list_skills(self):
        return [FakeSkill()]


class SkillRegistryForBadArgs(FakeSkillRegistry):
    """模拟的 Skill 注册表，额外提供 skill 发现提示和显式调用名称解析，用于测试错误参数修复流程。"""
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
    """模拟 Python readline 模块，用于测试 CLICompleter 的 Tab 补全行为。"""

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
    """模拟 TTY 终端，isatty() 总是返回 True。"""

    def isatty(self) -> bool:
        return True


class FakeDocument:
    """模拟 prompt_toolkit 的 Document，用于补全测试。"""

    def __init__(self, text_before_cursor: str) -> None:
        self.text_before_cursor = text_before_cursor


class FakePromptToolkitSession:
    """模拟 prompt_toolkit 的 PromptSession，记录 prompt_async 调用并返回固定回复。"""
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
    """CLI 运行时行为测试：覆盖命令识别、视图渲染、Agent 流式输出、工具调用修复、斜杠命令补全、思考动画、会话清空等。"""

    def test_exit_command_aliases(self) -> None:
        """验证退出命令的各种别名（exit/quit/退出 及对应斜杠形式）均能被正确识别，非退出命令不被误判。"""
        for value in ("exit", "quit", "/exit", "/quit", "退出", "/退出", " EXIT "):
            self.assertTrue(is_cli_exit_command(value))

        self.assertFalse(is_cli_exit_command("/memory"))

    def test_clear_command_aliases(self) -> None:
        """验证清空命令的各种别名（clear/清空 及对应斜杠形式）均能被正确识别，非清空命令不被误判。"""
        for value in ("clear", "/clear", "清空", "/清空", " CLEAR "):
            self.assertTrue(is_cli_clear_command(value))

        self.assertFalse(is_cli_clear_command("/clear-memory"))

    def test_cli_view_uses_compact_terminal_header_and_prompt(self) -> None:
        """验证 CLI 启动视图包含 Tulip Agent 品牌、模型信息、快捷帮助提示，且在非彩色模式下不含 ANSI 转义序列。"""
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
        """验证输入 ? 时打印包含所有快捷命令（clear/memory/skill/exit）的帮助信息。"""
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
        """验证 Agent 流式输出的每个 chunk 都通过 output_delta_sink 发送，最终回答和流式内容一致。"""
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
        """验证 model_with_streaming_enabled 使用 model_copy 创建独立副本而非修改原模型，且 streaming 标志正确开启。"""
        model = FakePydanticStyleStreamingModel()

        streamed_model = model_with_streaming_enabled(model)

        self.assertIsNot(streamed_model, model)
        self.assertFalse(model.streaming)
        self.assertTrue(streamed_model.streaming)

    def test_agent_streaming_preserves_tool_calls_before_final_answer(self) -> None:
        """验证 Agent 在工具调用之后继续流式输出最终回答，且最终回答正确到达 sink。"""
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
        """验证前置引导文本（如"我先查一下"）在工具调用前已被发送到 sink 并保留显示，即使后续抛出异常。"""
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
        """验证完整工具调用链路中前置引导文本和最终回答都被保留在流式输出中。"""
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
        """验证当模型调用 read_skill_file 缺少 relative_path 时，系统将错误反馈返回给模型，模型自动修正为 load_skill 调用。"""
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
        """验证当模型发出的工具调用缺少 name 字段时，系统将错误反馈返回给模型，模型改为直接回答。"""
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
        """验证 available_slash_command_specs 返回的列表中同时包含内置命令（/memory，/clear）和 skill 命令。"""
        commands = [
            item.command
            for item in available_slash_command_specs(FakeSkillRegistry())
        ]

        self.assertIn("/memory", commands)
        self.assertIn("/clear", commands)
        self.assertIn("/remember-procedure", commands)
        self.assertIn("/size-guide", commands)

    def test_prompt_toolkit_slash_matches_show_all_for_slash(self) -> None:
        """验证输入单独的 / 时显示所有可用斜杠命令，包括内置命令和 skill 命令。"""
        matches = prompt_toolkit_slash_matches(
            available_slash_command_specs(FakeSkillRegistry()),
            "/",
        )
        commands = [item.command for item in matches]

        self.assertIn("/memory", commands)
        self.assertIn("/remember", commands)
        self.assertIn("/size-guide", commands)

    def test_prompt_toolkit_slash_matches_filter_prefix_live(self) -> None:
        """验证斜杠命令补全按前缀实时过滤，且仅当光标位于行首斜杠位置时才生效，行中间不触发。"""
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
        """验证 PromptToolkitSlashCompleter 同步补全能根据前缀返回匹配项和正确的 start_position。"""
        completer = PromptToolkitSlashCompleter(
            available_slash_command_specs(FakeSkillRegistry())
        )

        completions = list(completer.get_completions(FakeDocument("/mem"), None))

        self.assertEqual([item.text for item in completions], ["/memory "])
        self.assertEqual(completions[0].start_position, -4)

    def test_prompt_toolkit_completer_supports_async_completion(self) -> None:
        """验证 PromptToolkitSlashCompleter 支持异步补全接口 get_completions_async。"""
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
        """验证在 TTY 终端下 CLIInputSession 使用 prompt_toolkit，并启用了实时补全（complete_while_typing）和斜杠命令补全器。"""
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
        """验证 readline 补全模式能够按 / 前缀匹配命令，临时移除 / 分隔符以支持 readline 的 Tab 补全，restore 后恢复分隔符。"""
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
        """验证思考状态指示器在 TTY 下正确显示动画（含动画帧、颜色 ANSI 码），并在停止时用 \\r\\033[K 清除行。"""
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
        """验证 CLIStatusEventSink 将检索相关事件（开始/找到/未找到）映射为正确的用户可读状态文本。"""
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
        """验证 readline 补全仅在行首（第一个词）时生效，非行首位置（如 "hello /rem"）不触发补全。"""
        readline = FakeReadline()
        completer = CLICompleter(available_slash_command_specs(FakeSkillRegistry()))
        readline.buffer = "hello /rem"
        readline.begin = 6

        with patch("rag_server.cli._readline", readline):
            self.assertIsNone(completer.complete("/rem", 0))

    def test_unknown_slash_command_prints_suggestions(self) -> None:
        """验证输入未知斜杠命令时打印未知命令提示，并给出相似命令的建议（如 /remeber 提示 /remember）。"""
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
        """验证已知的 skill 斜杠命令不会被未知命令拦截器误拦截，正常交由 Agent 处理。"""
        self.assertFalse(
            handle_unknown_slash_command(
                "/size-guide 帮我推荐尺码",
                skill_registry=FakeSkillRegistry(),
            )
        )

    def test_streaming_ignores_none_node_updates_after_final_answer(self) -> None:
        """验证流式输出最终回答后忽略 None 节点更新，不出现 'NoneType' 或错误日志等噪音输出。"""
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
        """验证多轮对话中工具调用的历史被正确保留，第二轮能感知到第一轮使用了工具。"""
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
        """验证 CLI 使用流式 token 输出且最终回答不会重复显示两次。"""
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
        """验证流式输出部分 token 后模型抛异常时，已输出的内容被保留，不显示错误信息或大模型调用失败提示。"""
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
        """验证流式传输中连接断开时，CLI 回退到 ainvoke 普通调用获取完整回答，且先清除不完整的流式输出。"""
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
        """验证当 sink 有输出但 graph 执行最终抛异常时，CLI 保留 sink 内容并显示完整错误信息和大模型调用失败提示。"""
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
        """验证 CLI 状态指示器在真实 RAG 检索流程中依次显示"分析问题"、"检索知识"、"找到知识"、"搜索知识库"等状态，不显示调试级别信息。"""
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
        """验证 clear 命令执行后，新一轮会话的消息历史长度为 1（仅包含新用户消息），上一轮历史被清空。"""
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
        """验证 format_cli_live_event 将工具调用事件格式化为包含 tool 名称和 args 的显示文本。"""
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
        """验证 retrieval tool 在 query_rewrite 关闭时回退到同步 search 方法，返回结果包含预期的 source 文件路径。"""
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
