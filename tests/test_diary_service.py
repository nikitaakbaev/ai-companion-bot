from datetime import date

from app.database.repositories import (
    create_diary_entry,
    get_or_create_chat,
    get_or_create_user,
    get_recent_diary_entries,
    save_message,
)
from app.llm.client import LLMResponseError
from app.memory.diary import DiaryService
from app.memory.schemas import DiaryEntryCreate, DiaryReflectionResult


class FakeSummarizer:
    def __init__(self, result: DiaryReflectionResult) -> None:
        self.result = result
        self.calls = 0

    async def summarize(self, messages, source_date):
        self.calls += 1
        return self.result


class FailingSummarizer:
    async def summarize(self, messages, source_date):
        raise LLMResponseError("HTTP 400")


async def add_message_set(session, count: int = 3):
    user = await get_or_create_user(session, 123, None, None, None)
    chat = await get_or_create_chat(session, 456, user.id, None, "private")
    for index in range(count):
        await save_message(
            session=session,
            chat_id=chat.id,
            user_id=user.id,
            role="user",
            text=f"message {index}",
            message_type="text",
            telegram_message_id=index,
        )
    return user


def make_service(result: DiaryReflectionResult, min_messages: int = 3) -> DiaryService:
    return DiaryService(
        summarizer=FakeSummarizer(result),
        min_messages=min_messages,
        max_messages=100,
        lookback_hours=24,
        skip_if_exists_for_date=True,
    )


def make_failing_service(min_messages: int = 3) -> DiaryService:
    return DiaryService(
        summarizer=FailingSummarizer(),
        min_messages=min_messages,
        max_messages=100,
        lookback_hours=24,
        skip_if_exists_for_date=True,
    )


async def test_diary_service_skips_when_not_enough_messages(session_factory) -> None:
    async with session_factory() as session:
        user = await add_message_set(session, count=1)
        result = await make_service(DiaryReflectionResult()).create_daily_summary(
            session,
            user.id,
            source_date=date(2026, 5, 16),
        )

        assert result.status == "skipped"
        assert result.skipped_reason == "not_enough_messages"


async def test_diary_service_skips_when_entries_exist(session_factory) -> None:
    async with session_factory() as session:
        user = await add_message_set(session, count=3)
        await create_diary_entry(
            session,
            user.id,
            title="Existing",
            content="Content",
            source_date=date(2026, 5, 16),
        )

        result = await make_service(DiaryReflectionResult()).create_daily_summary(
            session,
            user.id,
            source_date=date(2026, 5, 16),
        )

        assert result.status == "skipped"
        assert result.skipped_reason == "already_exists_for_date"


async def test_diary_service_saves_entries(session_factory) -> None:
    reflection = DiaryReflectionResult(
        day_summary="Summary",
        entries=[
            DiaryEntryCreate(
                title="Entry",
                content="Content",
                summary="Short",
                source_date=date(2026, 5, 16),
            )
        ],
    )
    async with session_factory() as session:
        user = await add_message_set(session, count=3)
        result = await make_service(reflection).create_daily_summary(
            session,
            user.id,
            source_date=date(2026, 5, 16),
        )
        entries = await get_recent_diary_entries(session, user.id, limit=10)

        assert result.status == "created"
        assert result.created_count == 1
        assert entries[0].title == "Entry"


async def test_diary_service_empty_when_llm_returns_no_entries(session_factory) -> None:
    async with session_factory() as session:
        user = await add_message_set(session, count=3)
        result = await make_service(DiaryReflectionResult(entries=[])).create_daily_summary(
            session,
            user.id,
            source_date=date(2026, 5, 16),
        )

        assert result.status == "empty"


async def test_diary_service_uses_local_fallback_when_llm_fails(session_factory) -> None:
    async with session_factory() as session:
        user = await add_message_set(session, count=3)
        result = await make_failing_service().create_daily_summary(
            session,
            user.id,
            source_date=date(2026, 5, 16),
        )
        entries = await get_recent_diary_entries(session, user.id, limit=10)

        assert result.status == "created"
        assert result.created_count == 1
        assert entries[0].title == "Conversation on 2026-05-16"
