from datetime import date

from app.database.models import DiaryEntry
from app.database.repositories import create_diary_entry, get_diary_entries_for_user
from app.llm.embeddings import EmbeddingService
from app.memory.rag import MemoryIndexer, build_diary_embedding_text
from app.memory.vector_store import VectorSearchResult, VectorStore


class FakeEmbeddingService(EmbeddingService):
    async def embed_text(self, text: str) -> list[float]:
        return [1.0, float(len(text))]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


class CapturingVectorStore(VectorStore):
    def __init__(self) -> None:
        self.upserts: list[dict] = []

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        text: str,
        metadata: dict,
    ) -> None:
        self.upserts.append(
            {"id": id, "embedding": embedding, "text": text, "metadata": metadata}
        )

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[VectorSearchResult]:
        return []

    async def delete(self, id: str) -> None:
        return None


def make_entry() -> DiaryEntry:
    return DiaryEntry(
        id=42,
        user_id=7,
        title="Local models",
        content="User runs LM Studio.",
        summary="LM Studio setup",
        facts_about_user=["Uses Windows"],
        facts_about_relationship=["Likes fast replies"],
        topics=["LLM", "Telegram"],
        importance=7,
        emotion="curious",
        source_date=date(2026, 5, 17),
    )


def test_build_diary_embedding_text() -> None:
    text = build_diary_embedding_text(make_entry())

    assert "Title: Local models" in text
    assert "Facts about user:" in text
    assert "- Uses Windows" in text
    assert "Importance: 7" in text


async def test_index_diary_entry_uses_stable_id_and_metadata() -> None:
    store = CapturingVectorStore()
    indexer = MemoryIndexer(FakeEmbeddingService(), store)

    embedding_id = await indexer.index_diary_entry(make_entry())

    assert embedding_id == "diary_entry:42"
    assert store.upserts[0]["id"] == "diary_entry:42"
    assert store.upserts[0]["metadata"]["user_id"] == 7
    assert store.upserts[0]["metadata"]["diary_entry_id"] == 42


async def test_reindex_user_diary_updates_embedding_id(session_factory) -> None:
    store = CapturingVectorStore()
    indexer = MemoryIndexer(FakeEmbeddingService(), store)
    async with session_factory() as session:
        await create_diary_entry(
            session=session,
            user_id=7,
            title="Plan",
            content="Build memory.",
            source_date=date(2026, 5, 17),
        )
        count = await indexer.reindex_user_diary(session, user_id=7)
        entries = await get_diary_entries_for_user(session, user_id=7)

    assert count == 1
    assert entries[0].embedding_id == f"diary_entry:{entries[0].id}"
