"""Telegram message handlers for the MVP."""

import logging
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message as TelegramMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.models import Chat, User
from app.database.repositories import get_or_create_chat, get_or_create_user, save_message

logger = logging.getLogger(__name__)
router = Router()

START_TEXT = (
    "Привет. Я AI companion bot.\n\n"
    "Сейчас я умею принимать сообщения и сохранять историю. На следующих этапах появятся "
    "LLM, память, дневник, RAG и проактивные сообщения."
)

HELP_TEXT = (
    "Доступные команды:\n\n"
    "/start — запустить бота\n"
    "/help — показать помощь\n"
    "/status — проверить состояние"
)

STATUS_TEXT = (
    "Бот работает.\n\n"
    "Этап: 2\n"
    "Telegram: подключен\n"
    "База данных: подключена\n"
    "LLM: ещё не подключена\n"
    "Память: ещё не подключена"
)

TEXT_STUB = (
    "Я получил сообщение и сохранил его в историю. "
    "На следующем этапе это сообщение будет передаваться в LLM."
)

ERROR_TEXT = "Произошла внутренняя ошибка при обработке сообщения."


async def _get_or_create_context(
    session: AsyncSession,
    message: TelegramMessage,
) -> tuple[User, Chat]:
    if message.from_user is None:
        raise ValueError("Telegram message has no from_user")

    user = await get_or_create_user(
        session=session,
        telegram_id=message.from_user.id,
        username=message.from_user.username,
        first_name=message.from_user.first_name,
        last_name=message.from_user.last_name,
    )
    chat = await get_or_create_chat(
        session=session,
        telegram_chat_id=message.chat.id,
        user_id=user.id,
        title=message.chat.title,
        chat_type=str(message.chat.type),
    )
    return user, chat


async def _save_interaction(
    session_factory: async_sessionmaker[AsyncSession],
    message: TelegramMessage,
    response_text: str,
) -> None:
    async with session_factory() as session:
        try:
            user, chat = await _get_or_create_context(session, message)
            await save_message(
                session=session,
                chat_id=chat.id,
                user_id=user.id,
                role="user",
                text=message.text,
                message_type="text",
                telegram_message_id=message.message_id,
            )

            response = await message.answer(response_text)

            await save_message(
                session=session,
                chat_id=chat.id,
                user_id=None,
                role="assistant",
                text=response_text,
                message_type="text",
                telegram_message_id=response.message_id,
            )
        except Exception:
            await session.rollback()
            raise


async def _handle_with_db(
    session_factory: async_sessionmaker[AsyncSession],
    message: TelegramMessage,
    response_text: str,
    log_message: str,
    log_context: dict[str, Any] | None = None,
) -> None:
    logger.info(log_message, extra=log_context or {})
    try:
        await _save_interaction(session_factory, message, response_text)
    except Exception:
        logger.exception("Failed to process Telegram message")
        await message.answer(ERROR_TEXT)


@router.message(Command("start"))
async def handle_start(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Handle /start."""
    await _handle_with_db(session_factory, message, START_TEXT, "Received /start command")


@router.message(Command("help"))
async def handle_help(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Handle /help."""
    await _handle_with_db(session_factory, message, HELP_TEXT, "Received /help command")


@router.message(Command("status"))
async def handle_status(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Handle /status."""
    await _handle_with_db(session_factory, message, STATUS_TEXT, "Received /status command")


@router.message(F.text.func(lambda text: not text.startswith("/")))
async def handle_text(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Handle any non-command text message."""
    await _handle_with_db(
        session_factory,
        message,
        TEXT_STUB,
        "Received text message",
        {"telegram_chat_id": message.chat.id},
    )
