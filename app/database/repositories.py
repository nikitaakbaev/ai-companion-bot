"""Database repositories."""

from datetime import date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.database.models import (
    AgentAction,
    AgentState,
    BotSettings,
    Chat,
    DiaryEntry,
    MediaFile,
    Message,
    User,
    utc_now,
)
from app.memory.schemas import DiaryEntryCreate


async def get_or_create_user(
    session: AsyncSession,
    telegram_id: int,
    username: str | None,
    first_name: str | None,
    last_name: str | None,
    language_code: str | None = None,
    is_bot: bool = False,
) -> User:
    """Create a user or update the existing Telegram profile fields."""
    result = await session.execute(select(User).where(User.telegram_id == telegram_id))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            language_code=language_code,
            is_bot=is_bot,
        )
        session.add(user)
    else:
        user.username = username
        user.first_name = first_name
        user.last_name = last_name
        user.language_code = language_code
        user.is_bot = is_bot
        user.updated_at = utc_now()

    await session.commit()
    await session.refresh(user)
    return user


async def get_or_create_chat(
    session: AsyncSession,
    telegram_chat_id: int,
    user_id: int | None,
    title: str | None,
    chat_type: str,
) -> Chat:
    """Create a chat or update its mutable Telegram fields."""
    result = await session.execute(
        select(Chat).where(Chat.telegram_chat_id == telegram_chat_id, Chat.user_id == user_id)
    )
    chat = result.scalar_one_or_none()

    if chat is None:
        chat = Chat(
            telegram_chat_id=telegram_chat_id,
            user_id=user_id,
            title=title,
            chat_type=chat_type,
        )
        session.add(chat)
    else:
        chat.title = title
        chat.chat_type = chat_type
        chat.updated_at = utc_now()

    await session.commit()
    await session.refresh(chat)
    return chat


def prepare_message_text(
    text: str | None,
    max_length: int | None,
    metadata_json: dict | None = None,
) -> tuple[str | None, dict | None]:
    """Truncate stored message text and annotate metadata when needed."""
    metadata = dict(metadata_json or {})
    if text is None or max_length is None or max_length <= 0 or len(text) <= max_length:
        return text, metadata or None

    metadata.update({"truncated": True, "original_length": len(text)})
    return text[:max_length], metadata


async def save_message(
    session: AsyncSession,
    chat_id: int,
    user_id: int | None,
    role: str,
    text: str | None,
    message_type: str,
    telegram_message_id: int | None,
    reply_to_message_id: int | None = None,
    media_file_id: int | None = None,
    metadata_json: dict | None = None,
    max_stored_message_length: int | None = None,
) -> Message:
    """Persist one message."""
    stored_text, stored_metadata = prepare_message_text(
        text,
        max_stored_message_length,
        metadata_json,
    )
    message = Message(
        chat_id=chat_id,
        user_id=user_id,
        role=role,
        text=stored_text,
        message_type=message_type,
        telegram_message_id=telegram_message_id,
        reply_to_message_id=reply_to_message_id,
        media_file_id=media_file_id,
        metadata_json=stored_metadata,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


async def get_recent_messages(
    session: AsyncSession,
    chat_id: int,
    limit: int = 20,
) -> list[Message]:
    """Return recent user/assistant messages in chronological order."""
    result = await session.execute(
        select(Message)
        .where(
            Message.chat_id == chat_id,
            Message.role.in_(("user", "assistant")),
            Message.text.is_not(None),
            Message.text != "",
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    )
    messages = list(result.scalars().all())
    return list(reversed(messages))


async def get_messages_for_period(
    session: AsyncSession,
    user_id: int,
    since: datetime,
    limit: int,
) -> list[Message]:
    """Return user/assistant messages for a user since a timestamp in chronological order."""
    result = await session.execute(
        select(Message)
        .join(Chat, Message.chat_id == Chat.id)
        .where(
            Chat.user_id == user_id,
            Message.role.in_(("user", "assistant")),
            Message.text.is_not(None),
            Message.text != "",
            Message.created_at >= since,
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    )
    messages = list(result.scalars().all())
    return list(reversed(messages))


async def save_agent_action(
    session: AsyncSession,
    user_id: int | None,
    chat_id: int | None,
    action_type: str,
    input_json: dict | None,
    output_json: dict | None,
    status: str = "success",
    error: str | None = None,
) -> AgentAction:
    """Persist one agent action."""
    action = AgentAction(
        user_id=user_id,
        chat_id=chat_id,
        action_type=action_type,
        input_json=input_json,
        output_json=output_json,
        status=status,
        error=error,
    )
    session.add(action)
    try:
        await session.commit()
        await session.refresh(action)
    except Exception:
        await session.rollback()
        raise
    return action


async def get_or_create_bot_settings(
    session: AsyncSession,
    user_id: int,
    settings: Settings,
) -> BotSettings:
    """Create default user settings or return existing settings."""
    existing = await get_bot_settings(session, user_id)
    if existing is not None:
        return await sync_bot_settings_from_env(session, existing, settings)

    bot_settings = BotSettings(
        user_id=user_id,
        character_name=settings.default_character_name,
        character_description=settings.default_character_description,
        personality_style=settings.default_personality_style,
        llm_model=settings.llm_model,
        vision_model=settings.vision_model,
        embedding_model=settings.embedding_model,
        proactive_enabled=settings.default_proactive_enabled,
        proactive_min_interval_minutes=settings.default_proactive_min_interval_minutes,
        proactive_max_interval_minutes=settings.default_proactive_max_interval_minutes,
        timezone=settings.default_timezone,
        silent_hours_start=settings.silent_hours_start,
        silent_hours_end=settings.silent_hours_end,
    )
    session.add(bot_settings)
    await session.commit()
    await session.refresh(bot_settings)
    return bot_settings


async def sync_bot_settings_from_env(
    session: AsyncSession,
    bot_settings: BotSettings,
    settings: Settings,
) -> BotSettings:
    """Sync env-driven defaults into persisted settings.

    Early stages do not have settings edit commands yet, so .env is treated as the source of truth for
    character/model defaults.
    """
    fields = {
        "character_name": settings.default_character_name,
        "character_description": settings.default_character_description,
        "personality_style": settings.default_personality_style,
        "llm_model": settings.llm_model,
        "vision_model": settings.vision_model,
        "embedding_model": settings.embedding_model,
        "proactive_enabled": settings.default_proactive_enabled,
        "proactive_min_interval_minutes": settings.default_proactive_min_interval_minutes,
        "proactive_max_interval_minutes": settings.default_proactive_max_interval_minutes,
        "timezone": settings.default_timezone,
        "silent_hours_start": settings.silent_hours_start,
        "silent_hours_end": settings.silent_hours_end,
    }
    changed = False
    for field, value in fields.items():
        if getattr(bot_settings, field) != value:
            setattr(bot_settings, field, value)
            changed = True

    if changed:
        bot_settings.updated_at = utc_now()
        await session.commit()
        await session.refresh(bot_settings)
    return bot_settings


async def get_bot_settings(session: AsyncSession, user_id: int) -> BotSettings | None:
    """Return user settings if they exist."""
    result = await session.execute(select(BotSettings).where(BotSettings.user_id == user_id))
    return result.scalar_one_or_none()


async def update_bot_settings(
    session: AsyncSession,
    user_id: int,
    **kwargs: Any,
) -> BotSettings:
    """Update user bot settings."""
    bot_settings = await get_bot_settings(session, user_id)
    if bot_settings is None:
        raise ValueError("BotSettings does not exist")

    for key, value in kwargs.items():
        if not hasattr(bot_settings, key):
            raise ValueError(f"Unknown BotSettings field: {key}")
        setattr(bot_settings, key, value)
    bot_settings.updated_at = utc_now()
    await session.commit()
    await session.refresh(bot_settings)
    return bot_settings


async def get_or_create_agent_state(session: AsyncSession, user_id: int) -> AgentState:
    """Create or return per-user agent state."""
    result = await session.execute(select(AgentState).where(AgentState.user_id == user_id))
    state = result.scalar_one_or_none()
    if state is not None:
        return state

    state = AgentState(user_id=user_id)
    session.add(state)
    await session.commit()
    await session.refresh(state)
    return state


async def update_agent_state(
    session: AsyncSession,
    user_id: int,
    **kwargs: Any,
) -> AgentState:
    """Update per-user agent state."""
    state = await get_or_create_agent_state(session, user_id)
    for key, value in kwargs.items():
        if not hasattr(state, key):
            raise ValueError(f"Unknown AgentState field: {key}")
        setattr(state, key, value)
    state.updated_at = utc_now()
    await session.commit()
    await session.refresh(state)
    return state


async def create_media_file(
    session: AsyncSession,
    user_id: int | None,
    chat_id: int | None,
    file_type: str,
    telegram_file_id: str | None = None,
    telegram_file_unique_id: str | None = None,
    mime_type: str | None = None,
    local_path: str | None = None,
    original_file_name: str | None = None,
    metadata_json: dict | None = None,
) -> MediaFile:
    """Create media file metadata."""
    media = MediaFile(
        user_id=user_id,
        chat_id=chat_id,
        telegram_file_id=telegram_file_id,
        telegram_file_unique_id=telegram_file_unique_id,
        file_type=file_type,
        mime_type=mime_type,
        local_path=local_path,
        original_file_name=original_file_name,
        metadata_json=metadata_json,
    )
    session.add(media)
    await session.commit()
    await session.refresh(media)
    return media


async def create_diary_entry(
    session: AsyncSession,
    user_id: int,
    title: str,
    content: str,
    summary: str | None = None,
    facts_about_user: list | None = None,
    facts_about_relationship: list | None = None,
    topics: list | None = None,
    importance: int = 5,
    emotion: str | None = None,
    source_date: date | None = None,
    embedding_id: str | None = None,
) -> DiaryEntry:
    """Create a diary entry prepared for stage 6."""
    entry = DiaryEntry(
        user_id=user_id,
        title=title,
        content=content,
        summary=summary,
        facts_about_user=facts_about_user,
        facts_about_relationship=facts_about_relationship,
        topics=topics,
        importance=importance,
        emotion=emotion,
        source_date=source_date,
        embedding_id=embedding_id,
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def diary_entries_exist_for_date(
    session: AsyncSession,
    user_id: int,
    source_date: date,
) -> bool:
    """Return whether diary entries already exist for a user and source date."""
    result = await session.execute(
        select(DiaryEntry.id)
        .where(DiaryEntry.user_id == user_id, DiaryEntry.source_date == source_date)
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def create_diary_entries(
    session: AsyncSession,
    user_id: int,
    entries: list[DiaryEntryCreate],
) -> list[DiaryEntry]:
    """Create multiple diary entries."""
    created = [
        DiaryEntry(
            user_id=user_id,
            title=entry.title,
            content=entry.content,
            summary=entry.summary,
            facts_about_user=entry.facts_about_user,
            facts_about_relationship=entry.facts_about_relationship,
            topics=entry.topics,
            importance=entry.importance,
            emotion=entry.emotion,
            source_date=entry.source_date,
        )
        for entry in entries
    ]
    session.add_all(created)
    try:
        await session.commit()
        for entry in created:
            await session.refresh(entry)
    except Exception:
        await session.rollback()
        raise
    return created


async def get_recent_diary_entries(
    session: AsyncSession,
    user_id: int,
    limit: int = 10,
) -> list[DiaryEntry]:
    """Return recent diary entries newest first."""
    result = await session.execute(
        select(DiaryEntry)
        .where(DiaryEntry.user_id == user_id)
        .order_by(DiaryEntry.created_at.desc(), DiaryEntry.id.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_recent_agent_actions(
    session: AsyncSession,
    user_id: int,
    limit: int = 10,
) -> list[AgentAction]:
    """Return recent agent actions newest first."""
    result = await session.execute(
        select(AgentAction)
        .where(AgentAction.user_id == user_id)
        .order_by(AgentAction.created_at.desc(), AgentAction.id.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def now_for_agent_state() -> datetime:
    """Return timestamp for agent state interaction fields."""
    return utc_now()
