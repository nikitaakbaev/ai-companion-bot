"""Diary memory service."""

import logging
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.repositories import (
    create_diary_entry,
    create_diary_entries,
    diary_entries_exist_for_date,
    get_messages_for_period,
    update_diary_entry_embedding_id,
)
from app.database.models import Message
from app.llm.client import LLMClientError
from app.llm.embeddings import EmbeddingError
from app.memory.schemas import DiaryEntryCreate, DiaryReflectionResult
from app.memory.summarizer import DiarySummarizer
from app.memory.vector_store import VectorStoreError

if TYPE_CHECKING:
    from app.memory.rag import MemoryIndexer

logger = logging.getLogger(__name__)


class DiaryServiceResult(BaseModel):
    """Result of one diary reflection run."""

    status: str
    created_count: int = 0
    indexed_count: int = 0
    indexing_failed: bool = False
    skipped_reason: str | None = None
    day_summary: str | None = None


class DiaryService:
    """Creates and reads long-term diary memories."""

    def __init__(
        self,
        summarizer: DiarySummarizer,
        min_messages: int,
        max_messages: int,
        lookback_hours: int,
        skip_if_exists_for_date: bool,
        memory_indexer: "MemoryIndexer | None" = None,
    ) -> None:
        self.summarizer = summarizer
        self.min_messages = min_messages
        self.max_messages = max_messages
        self.lookback_hours = lookback_hours
        self.skip_if_exists_for_date = skip_if_exists_for_date
        self.memory_indexer = memory_indexer

    async def create_daily_summary(
        self,
        session: AsyncSession,
        user_id: int,
        source_date: date | None = None,
    ) -> DiaryServiceResult:
        """Create diary entries from recent conversation history."""
        target_date = source_date or datetime.now(UTC).date()
        logger.info("Starting diary reflection", extra={"user_id": user_id})

        if self.skip_if_exists_for_date and await diary_entries_exist_for_date(
            session,
            user_id,
            target_date,
        ):
            logger.info("Diary reflection skipped: entries already exist")
            return DiaryServiceResult(status="skipped", skipped_reason="already_exists_for_date")

        since = datetime.now(UTC) - timedelta(hours=self.lookback_hours)
        messages = await get_messages_for_period(
            session=session,
            user_id=user_id,
            since=since,
            limit=self.max_messages,
        )
        logger.info("Found messages for diary reflection", extra={"message_count": len(messages)})

        if len(messages) < self.min_messages:
            logger.info("Diary reflection skipped: not enough messages")
            return DiaryServiceResult(status="skipped", skipped_reason="not_enough_messages")

        try:
            reflection = await self.summarizer.summarize(messages, target_date)
        except LLMClientError as exc:
            logger.warning("Diary reflection LLM unavailable; using local fallback: %s", exc)
            reflection = build_fallback_reflection(messages, target_date)
        if not reflection.entries:
            logger.info("Diary reflection produced no entries")
            return DiaryServiceResult(status="empty", day_summary=reflection.day_summary)

        created_entries = await create_diary_entries(session, user_id, reflection.entries)
        logger.info("Diary entries saved", extra={"created_count": len(created_entries)})
        indexed_count = 0
        indexing_failed = False
        if self.memory_indexer is not None:
            for entry in created_entries:
                try:
                    embedding_id = await self.memory_indexer.index_diary_entry(entry)
                    await update_diary_entry_embedding_id(session, entry.id, embedding_id)
                    indexed_count += 1
                except (EmbeddingError, VectorStoreError) as exc:
                    indexing_failed = True
                    logger.warning("Failed to index diary entry: %s", exc, extra={"entry_id": entry.id})
                except Exception:
                    indexing_failed = True
                    logger.exception("Failed to index diary entry", extra={"entry_id": entry.id})
        return DiaryServiceResult(
            status="created",
            created_count=len(created_entries),
            indexed_count=indexed_count,
            indexing_failed=indexing_failed,
            day_summary=reflection.day_summary,
        )

    async def remember_manual(
        self,
        session: AsyncSession,
        user_id: int,
        content: str,
        title: str | None = None,
        source_date: date | None = None,
        importance: int = 8,
        topics: list[str] | None = None,
    ) -> str:
        """Persist one explicit memory and index it when RAG is enabled."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content must not be empty")

        entry = await create_diary_entry(
            session=session,
            user_id=user_id,
            title=(title or "Manual memory").strip()[:500],
            content=clean_content,
            summary=clean_content[:500],
            facts_about_user=[clean_content],
            topics=topics or ["manual_memory"],
            importance=max(1, min(10, importance)),
            emotion="neutral",
            source_date=source_date or datetime.now(UTC).date(),
        )
        if self.memory_indexer is not None:
            try:
                embedding_id = await self.memory_indexer.index_diary_entry(entry)
                await update_diary_entry_embedding_id(session, entry.id, embedding_id)
            except (EmbeddingError, VectorStoreError) as exc:
                logger.warning("Failed to index manual memory: %s", exc, extra={"entry_id": entry.id})
        return f"diary_entry:{entry.id}"


def build_fallback_reflection(messages: list[Message], source_date: date) -> DiaryReflectionResult:
    """Create a minimal diary reflection when the LLM backend rejects the request."""
    useful_messages = [
        message
        for message in messages
        if message.text and message.role in {"user", "assistant"} and not message.text.startswith("/")
    ]
    if not useful_messages:
        return DiaryReflectionResult(entries=[], day_summary=None)

    last_messages = useful_messages[-12:]
    compact_lines = [
        f"{message.role}: {(message.text or '').replace(chr(10), ' ').strip()[:500]}"
        for message in last_messages
    ]
    user_lines = [
        (message.text or "").replace("\n", " ").strip()[:300]
        for message in last_messages
        if message.role == "user"
    ]
    content = "\n".join(compact_lines)
    day_summary = f"Local fallback summary for {source_date.isoformat()}."
    return DiaryReflectionResult(
        day_summary=day_summary,
        entries=[
            DiaryEntryCreate(
                title=f"Conversation on {source_date.isoformat()}",
                content=content,
                summary=day_summary,
                facts_about_user=user_lines[-5:],
                facts_about_relationship=[],
                topics=["conversation"],
                importance=5,
                emotion="neutral",
                source_date=source_date,
            )
        ],
    )
