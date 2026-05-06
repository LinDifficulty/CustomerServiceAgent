from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_server.memory_service import MemoryService


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            1.0 if "silk" in lower else 0.0,
            1.0 if "cotton" in lower else 0.0,
            1.0,
        ]


class MemoryServiceTest(unittest.TestCase):
    def test_search_uses_user_specific_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            try:
                for index in range(60):
                    memory.add_memory(f"other_{index}", "prefers silk")
                target = memory.add_memory("target_user", "prefers cotton")

                results = memory.search_memory("target_user", "silk", top_k=1)

                self.assertEqual([item["id"] for item in results], [target["id"]])
            finally:
                memory.close()

    def test_persists_separate_index_files_per_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            try:
                alice = memory.add_memory("alice", "prefers cotton")
                bob = memory.add_memory("bob", "prefers silk")
                alice_index_path, alice_ids_path = memory._user_index_paths("alice")
                bob_index_path, bob_ids_path = memory._user_index_paths("bob")
            finally:
                memory.close()

            self.assertTrue(alice_index_path.exists())
            self.assertTrue(alice_ids_path.exists())
            self.assertTrue(bob_index_path.exists())
            self.assertTrue(bob_ids_path.exists())
            self.assertEqual(len(list((Path(temp_dir) / "indexes").glob("*.faiss"))), 2)

            reloaded = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            try:
                alice_results = reloaded.search_memory("alice", "cotton", top_k=1)
                bob_results = reloaded.search_memory("bob", "silk", top_k=1)

                self.assertEqual([item["id"] for item in alice_results], [alice["id"]])
                self.assertEqual([item["id"] for item in bob_results], [bob["id"]])
            finally:
                reloaded.close()

    def test_forget_removes_only_affected_user_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            try:
                alice = memory.add_memory("alice", "prefers cotton")
                bob = memory.add_memory("bob", "prefers silk")

                self.assertTrue(memory.forget_memory(alice["id"], user_id="alice"))

                alice_results = memory.search_memory("alice", "cotton", top_k=1)
                bob_results = memory.search_memory("bob", "silk", top_k=1)

                self.assertEqual(alice_results, [])
                self.assertEqual([item["id"] for item in bob_results], [bob["id"]])
            finally:
                memory.close()

    def test_search_memory_layers_separates_profile_episode_and_procedure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = MemoryService(data_dir=temp_dir, embeddings=FakeEmbeddings())
            try:
                profile = memory.add_memory(
                    "user",
                    "prefers cotton",
                    memory_type="preference",
                )
                episode = memory.add_memory(
                    "user",
                    "last time asked about silk",
                    memory_type="episode",
                )
                procedure = memory.add_memory(
                    "user",
                    "always compare cotton and silk before answering",
                    memory_type="procedure",
                )

                layered = memory.search_memory_layers(
                    "user",
                    "cotton silk",
                    layer_top_k={"profile": 5, "episode": 5, "procedure": 5},
                )

                self.assertEqual([item["id"] for item in layered["profile"]], [profile["id"]])
                self.assertEqual([item["id"] for item in layered["episode"]], [episode["id"]])
                self.assertEqual(
                    [item["id"] for item in layered["procedure"]],
                    [procedure["id"]],
                )
                self.assertEqual(layered["procedure"][0]["memory_layer"], "procedure")
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
