"""RAG memory indexing and retrieval."""

import logging

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DiaryEntry
from app.database.repositories import (
    delete_diary_entries_for_user,
    get_diary_entries_for_user,
    update_diary_entry_embedding_id,
)
from app.llm.embeddings import EmbeddingService
from app.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_EMBEDDING_TEXT_CHARS = 6000


class RelevantMemory(BaseModel):
    """Relevant long-term memory returned by RAG."""

    diary_entry_id: int | None = None
    score: float
    title: str | None = None
    text: str
    metadata: dict = Field(default_factory=dict)


def build_diary_embedding_text(entry: DiaryEntry) -> str:
    """Build a stable semantic-search text for one diary entry."""
    lines: list[str] = []
    _append_field(lines, "Title", entry.title)
    _append_field(lines, "Summary", entry.summary)
    _append_field(lines, "Content", entry.content)
    _append_list(lines, "Facts about user", entry.facts_about_user or [])
    _append_list(lines, "Facts about relationship", entry.facts_about_relationship or [])
    _append_list(lines, "Topics", entry.topics or [])
    _append_field(lines, "Emotion", entry.emotion)
    _append_field(lines, "Importance", str(entry.importance) if entry.importance else None)
    return "\n".join(lines)[:MAX_EMBEDDING_TEXT_CHARS]


class MemoryIndexer:
    """Indexes diary entries into vector memory."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def index_diary_entry(self, entry: DiaryEntry) -> str:
        """Index one diary entry and return its stable embedding id."""
        embedding_text = build_diary_embedding_text(entry)
        embedding = await self.embedding_service.embed_text(embedding_text)
        embedding_id = f"diary_entry:{entry.id}"
        await self.vector_store.upsert(
            id=embedding_id,
            embedding=embedding,
            text=embedding_text,
            metadata=_build_diary_metadata(entry),
        )
        return embedding_id

    async def reindex_user_diary(
        self,
        session: AsyncSession,
        user_id: int,
    ) -> int:
        """Reindex all diary entries for a user."""
        entries = await get_diary_entries_for_user(session, user_id=user_id)
        indexed = 0
        for entry in entries:
            embedding_id = await self.index_diary_entry(entry)
            await update_diary_entry_embedding_id(session, entry.id, embedding_id)
            indexed += 1
        return indexed

    async def reset_user_memory(
        self,
        session: AsyncSession,
        user_id: int,
    ) -> int:
        """Delete a user's diary memory from vector store and database."""
        entries = await get_diary_entries_for_user(session, user_id=user_id)
        for entry in entries:
            vector_id = entry.embedding_id or f"diary_entry:{entry.id}"
            try:
                await self.vector_store.delete(vector_id)
            except Exception:
                logger.exception("Failed to delete memory vector", extra={"embedding_id": vector_id})
        await delete_diary_entries_for_user(session, user_id=user_id)
        return len(entries)


class MemoryRetriever:
    """Retrieves relevant memories from vector memory."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        top_k: int,
        min_score: float,
        max_context_chars: int,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_chars = max_context_chars

    async def retrieve(self, query: str, user_id: int) -> list[RelevantMemory]:
        """Retrieve relevant memories for a user query."""
        clean_query = query.strip()
        if not clean_query:
            return []

        embedding = await self.embedding_service.embed_text(clean_query)
        results = await self.vector_store.query(
            embedding=embedding,
            top_k=self.top_k,
            min_score=self.min_score,
        )

        memories: list[RelevantMemory] = []
        used_chars = 0
        for result in results:
            metadata = result.metadata
            if _metadata_int(metadata.get("user_id")) != user_id:
                continue
            if used_chars >= self.max_context_chars:
                break

            remaining = self.max_context_chars - used_chars
            text = result.text[:remaining]
            used_chars += len(text)
            memories.append(
                RelevantMemory(
                    diary_entry_id=_metadata_int(metadata.get("diary_entry_id")),
                    score=result.score,
                    title=metadata.get("title"),
                    text=text,
                    metadata=metadata,
                )
            )
        return memories


def _build_diary_metadata(entry: DiaryEntry) -> dict:
    return {
        "user_id": entry.user_id,
        "diary_entry_id": entry.id,
        "title": entry.title,
        "source_date": entry.source_date.isoformat() if entry.source_date else None,
        "importance": entry.importance,
        "topics": entry.topics or [],
    }


def _append_field(lines: list[str], label: str, value: str | None) -> None:
    if value:
        lines.append(f"{label}: {value}")


def _append_list(lines: list[str], label: str, values: list) -> None:
    clean_values = [str(value).strip() for value in values if str(value).strip()]
    if not clean_values:
        return
    lines.append(f"{label}:")
    lines.extend(f"- {value}" for value in clean_values)


def _metadata_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
