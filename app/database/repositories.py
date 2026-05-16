"""Database repositories."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Chat, Message, User, utc_now


async def get_or_create_user(
    session: AsyncSession,
    telegram_id: int,
    username: str | None,
    first_name: str | None,
    last_name: str | None,
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
        )
        session.add(user)
    else:
        user.username = username
        user.first_name = first_name
        user.last_name = last_name
        user.updated_at = utc_now()

    await session.commit()
    await session.refresh(user)
    return user


async def get_or_create_chat(
    session: AsyncSession,
    telegram_chat_id: int,
    user_id: int,
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


async def save_message(
    session: AsyncSession,
    chat_id: int,
    user_id: int | None,
    role: str,
    text: str | None,
    message_type: str,
    telegram_message_id: int | None,
) -> Message:
    """Persist one message."""
    message = Message(
        chat_id=chat_id,
        user_id=user_id,
        role=role,
        text=text,
        message_type=message_type,
        telegram_message_id=telegram_message_id,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message
