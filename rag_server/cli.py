from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .memory_service import LLMMemoryExtractor, MemoryService
from .mcp_service import DEFAULT_MCP_CONFIG_PATH, load_mcp_tools_from_config
from .query_rewrite import LLMQueryRewriter, search_with_query_rewrites
from .rag_service import RAGService
from .skill_service import SkillRegistry, build_skill_tools
from .trace_service import DEFAULT_TRACE_DIR, TraceRecorder, preview_text

DEFAULT_AGENT_MODEL = "qwen3-max-2026-01-23"
DEFAULT_USER_ID = "default_user"
DEFAULT_QUERY_REWRITE_MODE = "on"
QUERY_REWRITE_MODES = ("on", "off", "rewrite_only", "multi_query")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    latest_user_message: str
    memory_context: str
    skill_context: str
    active_skill_names: list[str]


def normalize_query_rewrite_mode(mode: str) -> str:
    """Convert the user-facing on/off switch into the internal strategy name."""
    return "multi_query" if mode == "on" else mode


def format_tongyi_error(
    model: ChatTongyi,
    messages: list[BaseMessage],
    error: Exception,
) -> str:
    """把 Tongyi 的异常转换成更可读的 CLI 错误信息。"""
    try:
        params = model._invocation_params(messages=messages, stop=None)
        response = model.client.call(**params)
        status_code = response.get("status_code")
        code = response.get("code")
        message = response.get("message")
        request_id = response.get("request_id")
        if status_code and code and message:
            return (
                "大模型调用失败。"
                f" status_code={status_code}, code={code}, message={message},"
                f" request_id={request_id}"
            )
    except Exception:
        pass

    return f"大模型调用失败：{error!r}"


def build_retrieval_tool(
    rag: RAGService,
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    rewrite_model_name: str = DEFAULT_AGENT_MODEL,
    trace_recorder: TraceRecorder | None = None,
):
    actual_mode = normalize_query_rewrite_mode(query_rewrite_mode)
    rewriter = (
        LLMQueryRewriter(
            model_name=rewrite_model_name,
            trace_recorder=trace_recorder,
        )
        if actual_mode != "off"
        else None
    )
    return build_retrieval_tool_with_rewrite(
        rag,
        query_rewrite_mode=actual_mode,
        rewriter=rewriter,
        trace_recorder=trace_recorder,
    )


def build_retrieval_tool_with_rewrite(
    rag: RAGService,
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    rewriter: LLMQueryRewriter | None = None,
    trace_recorder: TraceRecorder | None = None,
):
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)

    @tool(description="检索商品知识，返回与用户问题最相关的商品信息片段。")
    def search_product_knowledge(question: str) -> str:
        """检索商品知识库，返回与用户问题最相关的商品信息片段。"""
        trace_payload: dict[str, Any] = {
            "question": question,
            "query_rewrite_mode": actual_query_rewrite_mode,
        }
        if actual_query_rewrite_mode == "rewrite_only" and rewriter is not None:
            rewrite_result = rewriter.rewrite(question)
            trace_payload["rewritten_query"] = rewrite_result.rewritten_query
            trace_payload["retrieval_queries"] = [rewrite_result.rewritten_query]
            results = rag.search(
                query=rewrite_result.rewritten_query,
                top_k=3,
                candidate_top_k=10,
            )
        elif actual_query_rewrite_mode == "multi_query" and rewriter is not None:
            rewrite_result = rewriter.rewrite(question)
            retrieval_queries = [question, *rewrite_result.search_queries]
            deduplicated_queries = list(dict.fromkeys(retrieval_queries))
            trace_payload["rewritten_query"] = rewrite_result.rewritten_query
            trace_payload["retrieval_queries"] = deduplicated_queries
            results = search_with_query_rewrites(
                rag,
                original_query=question,
                rewritten_queries=deduplicated_queries,
                top_k=3,
                candidate_top_k=10,
                trace_recorder=trace_recorder,
            )
        else:
            trace_payload["retrieval_queries"] = [question]
            results = rag.search(
                query=question,
                top_k=3,
                candidate_top_k=10,
            )

        if not results:
            if trace_recorder is not None:
                trace_recorder.event(
                    "tool",
                    "tool.search_product_knowledge",
                    {**trace_payload, "result_count": 0},
                )
            return "未检索到相关商品知识。"

        blocks: list[str] = []
        for index, item in enumerate(results, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"片段{index}",
                        f"来源: {item['source']}",
                        f"内容: {item['content']}",
                    ]
                )
            )
        output = "\n\n".join(blocks)
        if trace_recorder is not None:
            trace_recorder.event(
                "tool",
                "tool.search_product_knowledge",
                {
                    **trace_payload,
                    "result_count": len(results),
                    "output_preview": preview_text(output),
                },
            )
        return output

    return search_product_knowledge


def coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return coerce_message_content(message.content).strip()
    return ""


def latest_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


MEMORY_LAYER_LABELS = {
    "profile": "用户画像与稳定偏好",
    "episode": "历史事件摘要",
    "procedure": "可复用流程记忆",
}


def format_memory_context(memories: list[dict]) -> str:
    if not memories:
        return ""

    blocks: list[str] = []
    for index, item in enumerate(memories, start=1):
        blocks.append(
            "\n".join(
                [
                    f"记忆{index}",
                    f"类型: {item['memory_type']}",
                    f"重要性: {item['importance']:.2f}",
                    f"内容: {item['content']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_layered_memory_context(layered_memories: dict[str, list[dict]]) -> str:
    blocks: list[str] = []
    for layer in ("profile", "episode", "procedure"):
        memories = layered_memories.get(layer) or []
        if not memories:
            continue

        layer_blocks = []
        for index, item in enumerate(memories, start=1):
            layer_blocks.append(
                "\n".join(
                    [
                        f"记忆{index}",
                        f"类型: {item['memory_type']}",
                        f"重要性: {item['importance']:.2f}",
                        f"内容: {item['content']}",
                    ]
                )
            )
        blocks.append(
            "\n".join(
                [
                    f"## {MEMORY_LAYER_LABELS.get(layer, layer)}",
                    *layer_blocks,
                ]
            )
        )
    return "\n\n".join(blocks)


def format_memory_list(memories: list[dict]) -> str:
    if not memories:
        return "当前没有长期记忆。"

    lines = []
    for item in memories:
        lines.append(
            f"{item['id'][:8]}  [{item.get('memory_layer', 'profile')}/"
            f"{item['memory_type']}] "
            f"{item['content']}  importance={item['importance']:.2f}"
        )
    return "\n".join(lines)


def resolve_memory_id(
    memory_service: MemoryService,
    user_id: str,
    memory_ref: str,
) -> tuple[str | None, str | None]:
    memories = memory_service.list_memories(user_id, limit=200)
    matches = [
        item["id"]
        for item in memories
        if item["id"] == memory_ref or item["id"].startswith(memory_ref)
    ]
    if not matches:
        return None, "没有找到这条记忆。"
    if len(matches) > 1:
        return None, "这个 ID 前缀匹配到多条记忆，请多输入几位。"
    return matches[0], None


def handle_memory_command(
    user_input: str,
    memory_service: MemoryService | None,
    user_id: str,
) -> bool:
    command, _, argument = user_input.partition(" ")
    if command not in {
        "/memory",
        "/remember",
        "/remember-procedure",
        "/remember-episode",
        "/forget",
        "/clear-memory",
    }:
        return False

    if memory_service is None:
        print("\n记忆系统当前已关闭。")
        return True

    try:
        if command == "/memory":
            print(f"\n{format_memory_list(memory_service.list_memories(user_id))}")
            return True

        if command in {"/remember", "/remember-procedure", "/remember-episode"}:
            content = argument.strip()
            if not content:
                print(f"\n用法: {command} 需要记住的内容")
                return True
            memory_type = {
                "/remember": "instruction",
                "/remember-procedure": "procedure",
                "/remember-episode": "episode",
            }[command]
            record = memory_service.add_memory(
                user_id,
                content,
                memory_type=memory_type,
                importance=0.8,
                source="manual",
            )
            print(f"\n已记住: {record['id'][:8]} {record['content']}")
            return True

        if command == "/forget":
            memory_ref = argument.strip()
            if not memory_ref:
                print("\n用法: /forget 记忆ID前缀")
                return True
            memory_id, error = resolve_memory_id(memory_service, user_id, memory_ref)
            if error is not None or memory_id is None:
                print(f"\n{error}")
                return True
            memory_service.forget_memory(memory_id, user_id=user_id)
            print("\n已删除这条记忆。")
            return True

        if command == "/clear-memory":
            count = memory_service.clear_user_memory(user_id)
            print(f"\n已清空 {count} 条长期记忆。")
            return True

    except Exception as error:
        print(f"\n记忆操作失败：{error!r}")
        return True

    return True


def build_tool_map(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """Build a tool lookup and fail early on duplicate tool names."""
    tool_map: dict[str, BaseTool] = {}
    duplicates: list[str] = []
    for item in tools:
        if item.name in tool_map:
            duplicates.append(item.name)
        tool_map[item.name] = item

    if duplicates:
        duplicate_names = ", ".join(sorted(set(duplicates)))
        raise ValueError(
            "Duplicate tool names detected. "
            f"Enable MCP tool_name_prefix or rename tools: {duplicate_names}"
        )
    return tool_map


def build_agent(
    rag: RAGService,
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    rewrite_model_name: str = DEFAULT_AGENT_MODEL,
    memory_service: MemoryService | None = None,
    memory_extractor: LLMMemoryExtractor | None = None,
    memory_top_k: int = 5,
    skills_enabled: bool = True,
    skill_registry: SkillRegistry | None = None,
    mcp_tools: list[BaseTool] | None = None,
    trace_recorder: TraceRecorder | None = None,
    agent_model: Any | None = None,
):
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)
    rewriter = None
    if actual_query_rewrite_mode != "off":
        rewriter = LLMQueryRewriter(
            model_name=rewrite_model_name,
            trace_recorder=trace_recorder,
        )

    retrieval_tool = build_retrieval_tool_with_rewrite(
        rag,
        query_rewrite_mode=actual_query_rewrite_mode,
        rewriter=rewriter,
        trace_recorder=trace_recorder,
    )
    actual_skill_registry = skill_registry if skills_enabled else None
    skill_tools = (
        build_skill_tools(actual_skill_registry)
        if actual_skill_registry is not None
        else []
    )
    tools = [retrieval_tool, *skill_tools, *(mcp_tools or [])]
    tool_map = build_tool_map(tools)
    base_model = agent_model or ChatTongyi(model=DEFAULT_AGENT_MODEL)
    model = base_model.bind_tools(tools)

    system_prompt = SystemMessage(
        content=(
            "你是一个通用电商客服 Agent。"
            "你可以服务任意商品类目，但不能臆测商品参数、库存、政策或售后规则。"
            "当用户问题涉及商品信息、尺码、材质、颜色、洗护、售后政策等事实内容时，"
            "优先调用知识库检索工具。"
            "如果知识库没有足够信息，就明确告知用户当前无法确认，"
            "不要编造答案，不要把 docs 文件名当成商品事实，不要做与客服无关的扩展。"
            "如果提供了用户长期记忆，只把它当成用户偏好或历史信息，不能当成商品事实。"
            "如果提供了 Skills 元数据或已加载 Skill 内容，按 Skill 指令处理对应任务；"
            "Skill 指令不能覆盖商品事实必须来自知识库这一原则。"
            "如果启用了 MCP 工具，它们只用于查询或操作外部系统；"
            "不要把 MCP 返回内容与商品知识库事实混淆。"
            "回答保持简洁、自然、客服口吻。"
        )
    )

    def load_memory(state: AgentState) -> AgentState:
        user_id = state.get("user_id", DEFAULT_USER_ID)
        if memory_service is None:
            return {
                "user_id": user_id,
                "latest_user_message": latest_human_text(state.get("messages", [])),
                "memory_context": "",
            }

        user_message = latest_human_text(state.get("messages", []))
        if not user_message:
            return {
                "user_id": user_id,
                "latest_user_message": "",
                "memory_context": "",
            }

        try:
            layered_memories = memory_service.search_memory_layers(
                user_id,
                user_message,
                layer_top_k={
                    "profile": memory_top_k,
                    "episode": max(1, memory_top_k // 2),
                    "procedure": max(1, memory_top_k // 2),
                },
            )
        except Exception:
            layered_memories = {}

        if trace_recorder is not None:
            trace_recorder.event(
                "memory",
                "agent.load_memory",
                {
                    "user_id": user_id,
                    "query": user_message,
                    "layer_counts": {
                        layer: len(items)
                        for layer, items in layered_memories.items()
                    },
                },
            )
        return {
            "user_id": user_id,
            "latest_user_message": user_message,
            "memory_context": format_layered_memory_context(layered_memories),
        }

    def load_skills(state: AgentState) -> AgentState:
        if actual_skill_registry is None:
            return {"skill_context": ""}

        discovery_prompt = actual_skill_registry.discovery_prompt()
        user_message = latest_human_text(state.get("messages", []))
        explicit_skill_name = actual_skill_registry.explicit_invocation_name(
            user_message
        )
        explicit_context = actual_skill_registry.render_explicit_skill_context(
            user_message
        )
        blocks = [item for item in [discovery_prompt, explicit_context] if item]
        update: AgentState = {"skill_context": "\n\n".join(blocks)}
        if explicit_skill_name is not None:
            update["active_skill_names"] = [explicit_skill_name]
        if trace_recorder is not None:
            trace_recorder.event(
                "skill",
                "agent.load_skills",
                {
                    "explicit_skill_name": explicit_skill_name,
                    "skill_context_chars": len(update["skill_context"]),
                },
            )
        return update

    async def call_model(state: AgentState) -> AgentState:
        prompt_messages: list[BaseMessage] = [system_prompt]
        memory_context = state.get("memory_context", "")
        if memory_context:
            prompt_messages.append(
                SystemMessage(
                    content=(
                        "以下是与当前用户问题可能相关的长期记忆。"
                        "这些内容已按 profile、episode、procedure 分层；"
                        "它们只代表用户偏好、约束、历史信息或可复用流程，不代表商品知识库事实；"
                        "只有与当前问题相关时才使用。\n\n"
                        f"{memory_context}"
                    )
                )
            )
        skill_context = state.get("skill_context", "")
        if skill_context:
            prompt_messages.append(SystemMessage(content=skill_context))

        if trace_recorder is not None:
            trace_recorder.event(
                "model",
                "agent.model_call_start",
                {
                    "prompt_message_count": len(prompt_messages),
                    "conversation_message_count": len(state["messages"]),
                },
            )
        response = await model.ainvoke([*prompt_messages, *state["messages"]])
        if trace_recorder is not None:
            trace_recorder.event(
                "model",
                "agent.model_call_end",
                {
                    "has_tool_calls": bool(
                        isinstance(response, AIMessage) and response.tool_calls
                    ),
                    "tool_call_count": (
                        len(response.tool_calls)
                        if isinstance(response, AIMessage) and response.tool_calls
                        else 0
                    ),
                    "content_preview": preview_text(
                        coerce_message_content(response.content)
                    ),
                },
            )
        return {"messages": [response]}

    def route_tools(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "save_memory"

    async def call_tools(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        tool_messages = []
        active_skill_names = list(state.get("active_skill_names", []))

        def allowed_tool_names() -> set[str] | None:
            if actual_skill_registry is None or not active_skill_names:
                return None

            names: set[str] = set()
            for skill_name in active_skill_names:
                skill = actual_skill_registry.get_skill(skill_name)
                if skill is None or not skill.allowed_tools:
                    continue
                names.update(skill.allowed_tools)

            if not names:
                return None
            return names | {"load_skill", "read_skill_file"}

        for tool_call in last_message.tool_calls:
            selected_tool = tool_map.get(tool_call["name"])
            if selected_tool is None:
                tool_result = f"未知工具: {tool_call['name']}"
            elif (
                allowed_tool_names() is not None
                and selected_tool.name not in allowed_tool_names()
            ):
                allowed = ", ".join(sorted(allowed_tool_names() or []))
                tool_result = (
                    f"当前已激活 skill 限制可用工具为: {allowed}。"
                    f"已拒绝调用: {selected_tool.name}"
                )
            else:
                if trace_recorder is not None:
                    trace_recorder.event(
                        "tool",
                        "agent.tool_call_start",
                        {
                            "tool_name": selected_tool.name,
                            "args": tool_call.get("args"),
                        },
                    )
                tool_result = await selected_tool.ainvoke(tool_call["args"])
                if trace_recorder is not None:
                    trace_recorder.event(
                        "tool",
                        "agent.tool_call_end",
                        {
                            "tool_name": selected_tool.name,
                            "result_preview": preview_text(tool_result),
                        },
                    )
                if selected_tool.name == "load_skill" and isinstance(
                    tool_call.get("args"),
                    dict,
                ):
                    loaded_skill_name = str(
                        tool_call["args"].get("name") or ""
                    ).strip().lower()
                    if (
                        actual_skill_registry is not None
                        and actual_skill_registry.get_skill(loaded_skill_name)
                        is not None
                        and loaded_skill_name not in active_skill_names
                    ):
                        active_skill_names.append(loaded_skill_name)
            tool_messages.append(
                {
                    "role": "tool",
                    "content": (
                        tool_result
                        if isinstance(tool_result, str | list)
                        else str(tool_result)
                    ),
                    "tool_call_id": tool_call["id"],
                }
            )
        return {
            "messages": tool_messages,
            "active_skill_names": active_skill_names,
        }

    def save_memory(state: AgentState) -> AgentState:
        if memory_service is None or memory_extractor is None:
            return {}

        messages = state.get("messages", [])
        if not messages:
            return {}

        final_message = latest_ai_message(messages)
        if final_message is None or final_message.tool_calls:
            if trace_recorder is not None:
                trace_recorder.event(
                    "memory",
                    "agent.save_memory_skipped",
                    {"reason": "no_final_ai_message_or_pending_tool_calls"},
                )
            return {}

        user_id = state.get("user_id", DEFAULT_USER_ID)
        user_message = (
            state.get("latest_user_message", "").strip()
            or latest_human_text(messages)
        )
        assistant_message = coerce_message_content(final_message.content).strip()
        if not user_message or not assistant_message:
            if trace_recorder is not None:
                trace_recorder.event(
                    "memory",
                    "agent.save_memory_skipped",
                    {
                        "reason": "empty_user_or_assistant_message",
                        "has_user_message": bool(user_message),
                        "has_assistant_message": bool(assistant_message),
                    },
                )
            return {}

        try:
            try:
                existing_memories = memory_service.search_memory(
                    user_id,
                    user_message,
                    top_k=8,
                )
            except Exception as error:
                existing_memories = []
                if trace_recorder is not None:
                    trace_recorder.event(
                        "memory",
                        "agent.search_existing_memory_failed",
                        {"user_id": user_id, "error": repr(error)},
                        level="warning",
                    )
            extracted_memories = memory_extractor.extract(
                user_message=user_message,
                assistant_message=assistant_message,
                existing_memories=existing_memories,
            )

            if not extracted_memories:
                if trace_recorder is not None:
                    trace_recorder.event(
                        "memory",
                        "agent.save_memory_skipped",
                        {"reason": "extractor_returned_empty", "user_id": user_id},
                    )
                return {}

            known_contents = {
                item["content"]
                for item in memory_service.list_memories(user_id, limit=200)
            }
            new_memories = [
                {
                    "content": item.content,
                    "memory_type": item.memory_type,
                    "importance": item.importance,
                    "source": "conversation",
                    "expires_at": item.expires_at,
                }
                for item in extracted_memories
                if item.content not in known_contents
            ]
            if new_memories:
                memory_service.add_memories(user_id, new_memories)
                if trace_recorder is not None:
                    trace_recorder.event(
                        "memory",
                        "agent.save_memory",
                        {
                            "user_id": user_id,
                            "new_memory_count": len(new_memories),
                            "memory_types": [
                                item["memory_type"] for item in new_memories
                            ],
                        },
                    )
        except Exception as error:
            if trace_recorder is not None:
                trace_recorder.event(
                    "memory",
                    "agent.save_memory_failed",
                    {"user_id": user_id, "error": repr(error)},
                    level="error",
                )
            return {}

        return {}

    graph = StateGraph(AgentState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("load_skills", load_skills)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("save_memory", save_memory)
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "load_skills")
    graph.add_edge("load_skills", "agent")
    graph.add_conditional_edges(
        "agent",
        route_tools,
        {"tools": "tools", "save_memory": "save_memory"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("save_memory", END)
    return graph.compile(), base_model, system_prompt


async def run_cli_async(
    *,
    query_rewrite_mode: str = DEFAULT_QUERY_REWRITE_MODE,
    rewrite_model_name: str = DEFAULT_AGENT_MODEL,
    bm25_enabled: bool = True,
    cross_encoder_enabled: bool = True,
    user_id: str = DEFAULT_USER_ID,
    memory_enabled: bool = True,
    memory_model_name: str = DEFAULT_AGENT_MODEL,
    skills_enabled: bool = True,
    skill_dirs: list[str] | None = None,
    mcp_enabled: bool = False,
    mcp_config_path: str = DEFAULT_MCP_CONFIG_PATH,
    trace_enabled: bool = False,
    trace_dir: str = DEFAULT_TRACE_DIR,
) -> None:
    actual_query_rewrite_mode = normalize_query_rewrite_mode(query_rewrite_mode)
    trace_recorder = (
        TraceRecorder(
            trace_dir=trace_dir,
            default_tags={"entrypoint": "cli", "user_id": user_id},
        )
        if trace_enabled
        else None
    )
    rag = RAGService(
        data_dir="data",
        default_use_bm25=bm25_enabled,
        default_use_rerank=cross_encoder_enabled,
        trace_recorder=trace_recorder,
    )
    memory_service = MemoryService(data_dir="memory") if memory_enabled else None
    memory_extractor = (
        LLMMemoryExtractor(model_name=memory_model_name)
        if memory_service is not None
        else None
    )
    skill_registry = (
        SkillRegistry.from_project_root(
            Path.cwd(),
            extra_skill_dirs=skill_dirs,
        )
        if skills_enabled
        else None
    )
    mcp_result = (
        await load_mcp_tools_from_config(mcp_config_path)
        if mcp_enabled
        else None
    )
    mcp_tools = mcp_result.tools if mcp_result is not None else []
    app, model, system_prompt = build_agent(
        rag,
        query_rewrite_mode=actual_query_rewrite_mode,
        rewrite_model_name=rewrite_model_name,
        memory_service=memory_service,
        memory_extractor=memory_extractor,
        skills_enabled=skills_enabled,
        skill_registry=skill_registry,
        mcp_tools=mcp_tools,
        trace_recorder=trace_recorder,
    )
    messages: list[BaseMessage] = []

    print("电商客服 Agent 已启动，输入 quit 或 exit 结束。")
    print(f"当前 query 改写模式: {actual_query_rewrite_mode}")
    print(f"当前 BM25 模式: {'on' if bm25_enabled else 'off'}")
    print(f"当前 CrossEncoder 精排模式: {'on' if cross_encoder_enabled else 'off'}")
    print(f"当前 memory 模式: {'on' if memory_enabled else 'off'}")
    if trace_recorder is not None:
        print(f"当前 trace 模式: on ({trace_recorder.path})")
    else:
        print("当前 trace 模式: off")
    if skill_registry is not None:
        skills = skill_registry.list_skills()
        print(f"当前 skills 模式: on ({len(skills)} 个)")
        if skills:
            print("可用 skills: " + ", ".join(skill.name for skill in skills))
        if skill_registry.errors:
            print("skills 加载警告:")
            for error in skill_registry.errors:
                print(f"- {error}")
    else:
        print("当前 skills 模式: off")
    if mcp_result is not None:
        server_names = ", ".join(mcp_result.server_names) or "无"
        tool_names = ", ".join(tool.name for tool in mcp_tools) or "无"
        print(f"当前 MCP 模式: on ({len(mcp_tools)} 个工具)")
        print(f"MCP servers: {server_names}")
        print(f"MCP tools: {tool_names}")
    else:
        print("当前 MCP 模式: off")
    print(f"当前 user_id: {user_id}")
    try:
        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("客服会话已结束。")
                break
            if not user_input:
                continue
            if handle_memory_command(user_input, memory_service, user_id):
                continue

            input_messages = messages + [HumanMessage(content=user_input)]
            try:
                result = await app.ainvoke(
                    {
                        "messages": input_messages,
                        "user_id": user_id,
                    }
                )
            except Exception as error:
                model_messages = [system_prompt, *input_messages]
                print(f"\n客服: {format_tongyi_error(model, model_messages, error)}")
                continue

            messages = result["messages"]

            final_message = messages[-1]
            if isinstance(final_message, AIMessage):
                print(f"\n客服: {final_message.content}")
    finally:
        if memory_service is not None:
            memory_service.close()


def run_cli(**kwargs: Any) -> None:
    asyncio.run(run_cli_async(**kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ecommerce customer-service CLI.")
    parser.add_argument(
        "--query-rewrite",
        choices=QUERY_REWRITE_MODES,
        default=DEFAULT_QUERY_REWRITE_MODE,
        help=(
            "Control retrieval query rewriting. "
            "'on' is an alias for 'multi_query'. Defaults to on."
        ),
    )
    parser.add_argument(
        "--bm25",
        choices=["on", "off"],
        default="on",
        help="Enable or disable BM25 keyword retrieval. Defaults to on.",
    )
    parser.add_argument(
        "--cross-encoder",
        choices=["on", "off"],
        default="off",
        help="Enable or disable CrossEncoder reranking. Defaults to on.",
    )
    parser.add_argument(
        "--rewrite-model",
        default=DEFAULT_AGENT_MODEL,
        help="Model name used by the query rewriter. Defaults to the agent model.",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help="User id used to scope long-term memories.",
    )
    parser.add_argument(
        "--memory",
        choices=["on", "off"],
        default="on",
        help="Enable or disable long-term memory.",
    )
    parser.add_argument(
        "--memory-model",
        default=DEFAULT_AGENT_MODEL,
        help="Model name used by the memory extractor. Defaults to the agent model.",
    )
    parser.add_argument(
        "--skills",
        choices=["on", "off"],
        default="on",
        help="Enable or disable Anthropic-style skills. Defaults to on.",
    )
    parser.add_argument(
        "--skills-dir",
        action="append",
        default=None,
        help=(
            "Additional Anthropic-style skills directory. "
            "Can be passed multiple times. Defaults to .claude/skills."
        ),
    )
    parser.add_argument(
        "--mcp",
        choices=["on", "off"],
        default="off",
        help="Enable or disable MCP client tools. Defaults to off.",
    )
    parser.add_argument(
        "--mcp-config",
        default=DEFAULT_MCP_CONFIG_PATH,
        help=(
            "Path to MCP server JSON config. "
            f"Defaults to {DEFAULT_MCP_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--trace",
        choices=["on", "off"],
        default="off",
        help="Enable or disable JSONL runtime tracing. Defaults to off.",
    )
    parser.add_argument(
        "--trace-dir",
        default=DEFAULT_TRACE_DIR,
        help=f"Directory for JSONL trace files. Defaults to {DEFAULT_TRACE_DIR}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cli(
        query_rewrite_mode=args.query_rewrite,
        rewrite_model_name=args.rewrite_model,
        bm25_enabled=args.bm25 == "on",
        cross_encoder_enabled=args.cross_encoder == "on",
        user_id=args.user_id,
        memory_enabled=args.memory == "on",
        memory_model_name=args.memory_model,
        skills_enabled=args.skills == "on",
        skill_dirs=args.skills_dir,
        mcp_enabled=args.mcp == "on",
        mcp_config_path=args.mcp_config,
        trace_enabled=args.trace == "on",
        trace_dir=args.trace_dir,
    )


if __name__ == "__main__":
    main()
