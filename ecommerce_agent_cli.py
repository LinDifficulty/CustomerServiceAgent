from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from rag_service import RAGService


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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


def build_retrieval_tool(rag: RAGService):
    @tool(description="检索商品知识，返回与用户问题最相关的商品信息片段。")
    def search_product_knowledge(question: str) -> str:
        """检索商品知识库，返回与用户问题最相关的商品信息片段。"""
        results = rag.search(
            query=question,
            top_k=3,
            use_rerank=True,
            candidate_top_k=10,
        )
        if not results:
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
        return "\n\n".join(blocks)

    return search_product_knowledge


def build_agent(rag: RAGService):
    retrieval_tool = build_retrieval_tool(rag)
    tools = [retrieval_tool]
    model = ChatTongyi(model="qwen3-max-2026-01-23").bind_tools(tools)

    system_prompt = SystemMessage(
        content=(
            "你是一个通用电商客服 Agent。"
            "你可以服务任意商品类目，但不能臆测商品参数、库存、政策或售后规则。"
            "当用户问题涉及商品信息、尺码、材质、颜色、洗护、售后政策等事实内容时，"
            "优先调用知识库检索工具。"
            "如果知识库没有足够信息，就明确告知用户当前无法确认，"
            "不要编造答案，不要把 docs 文件名当成商品事实，不要做与客服无关的扩展。"
            "回答保持简洁、自然、客服口吻。"
        )
    )

    def call_model(state: AgentState) -> AgentState:
        response = model.invoke([system_prompt, *state["messages"]])
        return {"messages": [response]}

    def route_tools(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    def call_tools(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        tool_messages = []
        for tool_call in last_message.tool_calls:
            if tool_call["name"] != retrieval_tool.name:
                continue
            tool_result = retrieval_tool.invoke(tool_call["args"])
            tool_messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"],
                }
            )
        return {"messages": tool_messages}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(), model, system_prompt


def run_cli() -> None:
    rag = RAGService(data_dir="data")
    app, model, system_prompt = build_agent(rag)
    messages: list[BaseMessage] = []

    print("电商客服 Agent 已启动，输入 quit 或 exit 结束。")
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("客服会话已结束。")
            break
        if not user_input:
            continue

        input_messages = messages + [HumanMessage(content=user_input)]
        try:
            result = app.invoke({"messages": input_messages})
        except Exception as error:
            model_messages = [system_prompt, *input_messages]
            print(f"\n客服: {format_tongyi_error(model, model_messages, error)}")
            continue

        messages = result["messages"]

        final_message = messages[-1]
        if isinstance(final_message, AIMessage):
            print(f"\n客服: {final_message.content}")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
