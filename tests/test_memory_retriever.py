from app.llm.embeddings import EmbeddingService
from app.memory.rag import MemoryRetriever
from app.memory.vector_store import VectorSearchResult, VectorStore


class FakeEmbeddingService(EmbeddingService):
    async def embed_text(self, text: str) -> list[float]:
        return [1.0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class StaticVectorStore(VectorStore):
    async def upsert(
        self,
        id: str,
        embedding: list[float],
        text: str,
        metadata: dict,
    ) -> None:
        return None

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                id="a",
                score=0.9,
                text="a" * 20,
                metadata={"user_id": 1, "diary_entry_id": 10, "title": "A"},
            ),
            VectorSearchResult(
                id="b",
                score=0.8,
                text="other user",
                metadata={"user_id": 2, "diary_entry_id": 20, "title": "B"},
            ),
        ]

    async def delete(self, id: str) -> None:
        return None


async def test_retriever_returns_empty_for_empty_query() -> None:
    retriever = MemoryRetriever(FakeEmbeddingService(), StaticVectorStore(), 5, 0.65, 100)

    assert await retriever.retrieve("", user_id=1) == []


async def test_retriever_filters_by_user_and_limits_context() -> None:
    retriever = MemoryRetriever(FakeEmbeddingService(), StaticVectorStore(), 5, 0.65, 8)

    memories = await retriever.retrieve("query", user_id=1)

    assert len(memories) == 1
    assert memories[0].diary_entry_id == 10
    assert memories[0].text == "a" * 8
