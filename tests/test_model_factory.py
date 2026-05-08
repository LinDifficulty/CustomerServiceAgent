from __future__ import annotations

import unittest

from rag_server.model_factory import create_chat_model, create_embeddings, create_reranker


class FakeChatProvider:
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature


class FakeEmbeddingProvider:
    def __init__(self, model_name: str, dimensions: int = 3) -> None:
        self.model_name = model_name
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self.dimensions for _ in texts]


class FakeRerankerProvider:
    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device

    def predict(self, pairs, **kwargs):
        return [1.0 for _ in pairs]


class ModelFactoryTests(unittest.TestCase):
    def test_custom_chat_provider_import_path(self) -> None:
        model = create_chat_model(
            f"{__name__}.FakeChatProvider",
            "chat-model",
            temperature=0.2,
        )

        self.assertEqual(model.model, "chat-model")
        self.assertEqual(model.temperature, 0.2)

    def test_custom_embedding_provider_import_path(self) -> None:
        embeddings = create_embeddings(
            f"{__name__}:FakeEmbeddingProvider",
            "embedding-model",
            dimensions=5,
        )

        self.assertEqual(embeddings.model_name, "embedding-model")
        self.assertEqual(embeddings.embed_documents(["a"])[0], [1.0] * 5)

    def test_custom_reranker_provider_import_path(self) -> None:
        reranker = create_reranker(
            f"{__name__}.FakeRerankerProvider",
            "reranker-model",
            device="cpu",
        )

        self.assertEqual(reranker.model_name_or_path, "reranker-model")
        self.assertEqual(reranker.device, "cpu")


if __name__ == "__main__":
    unittest.main()
