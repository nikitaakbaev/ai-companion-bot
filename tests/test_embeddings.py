import pytest

from app.llm.embeddings import (
    EmbeddingError,
    EmbeddingService,
    HashEmbeddingService,
    SentenceTransformerEmbeddingService,
)


class FakeEmbeddingService(EmbeddingService):
    async def embed_text(self, text: str) -> list[float]:
        return [float(len(text))]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


async def test_sentence_transformer_rejects_empty_text() -> None:
    service = SentenceTransformerEmbeddingService("unused")

    with pytest.raises(EmbeddingError):
        await service.embed_text("")


async def test_embedding_service_returns_one_embedding_per_text() -> None:
    service = FakeEmbeddingService()

    embeddings = await service.embed_texts(["one", "three"])

    assert embeddings == [[3.0], [5.0]]


async def test_hash_embedding_service_is_deterministic_and_normalized() -> None:
    service = HashEmbeddingService(dimensions=16)

    first = await service.embed_text("Привет memory")
    second = await service.embed_text("Привет memory")

    assert first == second
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1.0)


async def test_sentence_transformer_falls_back_when_model_load_fails() -> None:
    service = SentenceTransformerEmbeddingService("unused", fallback_service=HashEmbeddingService(8))
    service._load_error = "broken"

    embedding = await service.embed_text("hello")

    assert len(embedding) == 8
