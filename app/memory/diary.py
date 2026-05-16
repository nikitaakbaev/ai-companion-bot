"""Diary memory service."""

import logging
from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.repositories import (
    create_diary_entries,
    diary_entries_exist_for_date,
    get_messages_for_period,
)
from app.memory.summarizer import DiarySummarizer

logger = logging.getLogger(__name__)


class DiaryServiceResult(BaseModel):
    """Result of one diary reflection run."""

    status: str
    created_count: int = 0
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
    ) -> None:
        self.summarizer = summarizer
        self.min_messages = min_messages
        self.max_messages = max_messages
        self.lookback_hours = lookback_hours
        self.skip_if_exists_for_date = skip_if_exists_for_date

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

        reflection = await self.summarizer.summarize(messages, target_date)
        if not reflection.entries:
            logger.info("Diary reflection produced no entries")
            return DiaryServiceResult(status="empty", day_summary=reflection.day_summary)

        created_entries = await create_diary_entries(session, user_id, reflection.entries)
        logger.info("Diary entries saved", extra={"created_count": len(created_entries)})
        return DiaryServiceResult(
            status="created",
            created_count=len(created_entries),
            day_summary=reflection.day_summary,
        )
