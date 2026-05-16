import pytest
from sqlalchemy import select

from app.database.models import Chat, Message, User
from app.database.repositories import get_or_create_chat, get_or_create_user, save_message
from app.database.session import create_engine_from_url, create_session_factory, init_db


@pytest.fixture
async def session_factory():
    engine = create_engine_from_url("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    factory = create_session_factory(engine)
    try:
        yield factory
    finally:
        await engine.dispose()


async def test_get_or_create_user_creates_user(session_factory) -> None:
    async with session_factory() as session:
        user = await get_or_create_user(
            session=session,
            telegram_id=123,
            username="user",
            first_name="First",
            last_name="Last",
        )

        assert user.id is not None
        assert user.telegram_id == 123
        assert user.username == "user"


async def test_get_or_create_user_does_not_duplicate(session_factory) -> None:
    async with session_factory() as session:
        first = await get_or_create_user(session, 123, "old", "Old", None)
        second = await get_or_create_user(session, 123, "new", "New", "Name")

        result = await session.execute(select(User).where(User.telegram_id == 123))
        users = result.scalars().all()

        assert first.id == second.id
        assert len(users) == 1
        assert second.username == "new"
        assert second.first_name == "New"
        assert second.last_name == "Name"


async def test_get_or_create_chat_creates_chat(session_factory) -> None:
    async with session_factory() as session:
        user = await get_or_create_user(session, 123, None, None, None)
        chat = await get_or_create_chat(
            session=session,
            telegram_chat_id=456,
            user_id=user.id,
            title="Chat",
            chat_type="private",
        )

        result = await session.execute(select(Chat).where(Chat.telegram_chat_id == 456))

        assert chat.id is not None
        assert result.scalar_one().title == "Chat"


async def test_save_message_persists_message(session_factory) -> None:
    async with session_factory() as session:
        user = await get_or_create_user(session, 123, None, None, None)
        chat = await get_or_create_chat(session, 456, user.id, None, "private")
        message = await save_message(
            session=session,
            chat_id=chat.id,
            user_id=user.id,
            role="user",
            text="hello",
            message_type="text",
            telegram_message_id=789,
        )

        result = await session.execute(select(Message).where(Message.id == message.id))
        saved = result.scalar_one()

        assert saved.text == "hello"
        assert saved.role == "user"
        assert saved.telegram_message_id == 789
