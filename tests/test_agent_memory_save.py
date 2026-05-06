from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from rag_server.cli import build_agent
from rag_server.memory_service import ExtractedMemory, MemoryService
from rag_server.rag_service import RAGService


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            1.0 if "black" in lower or "黑色" in lower else 0.0,
            1.0,
        ]


class FakeModel:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="好的，已记住您偏好黑色通勤风格。")


class FakeMemoryExtractor:
    def extract(self, *, user_message, assistant_message, existing_memories=None):
        return [
            ExtractedMemory(
                content="喜欢黑色通勤风格",
                memory_type="preference",
                importance=0.8,
            )
        ]


class AgentMemorySaveTest(unittest.TestCase):
    def test_agent_saves_memory_from_finished_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rag = RAGService(
                data_dir=str(Path(temp_dir) / "data"),
                embeddings=FakeEmbeddings(),
                default_use_rerank=False,
            )
            memory = MemoryService(
                data_dir=str(Path(temp_dir) / "memory"),
                embeddings=FakeEmbeddings(),
            )
            try:
                app, _, _ = build_agent(
                    rag,
                    query_rewrite_mode="off",
                    memory_service=memory,
                    memory_extractor=FakeMemoryExtractor(),
                    skills_enabled=False,
                    agent_model=FakeModel(),
                )

                asyncio.run(
                    app.ainvoke(
                        {
                            "messages": [
                                HumanMessage(content="请记住我喜欢黑色通勤风格")
                            ],
                            "user_id": "user",
                        }
                    )
                )

                memories = memory.list_memories("user")
                self.assertEqual(len(memories), 1)
                self.assertEqual(memories[0]["content"], "喜欢黑色通勤风格")
                self.assertEqual(memories[0]["memory_layer"], "profile")
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
