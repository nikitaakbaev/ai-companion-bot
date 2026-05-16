"""Telegram message handlers for the MVP."""

import logging
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message as TelegramMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.orchestrator import AgentOrchestrator
from app.config import Settings
from app.database.models import Chat, User
from app.database.repositories import (
    get_or_create_chat,
    get_or_create_user,
    get_recent_messages,
    save_message,
)
from app.llm.client import ChatMessage, LLMClientError

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
    "Этап: 3\n"
    "Telegram: подключен\n"
    "База данных: подключена\n"
    "LLM: подключена через OpenAI-compatible API\n"
    "Модель: {model}\n"
    "Память: ещё не подключена\n"
    "RAG: ещё не подключен\n"
    "Vision: ещё не подключен"
)

ERROR_TEXT = "Произошла внутренняя ошибка при обработке сообщения."
LLM_ERROR_TEXT = (
    "Сейчас я не могу получить ответ от LLM. "
    "Проверь, что LM Studio/Ollama запущен и модель загружена."
)
LLM_TEST_ERROR_TEXT = (
    "LLM недоступна. Проверь LLM_BASE_URL, LLM_MODEL и запущенный сервер модели."
)
LLM_EMPTY_RESPONSE_TEXT = (
    "LLM подключена, но вернула пустой ответ. "
    "Проверь выбранную модель и попробуй увеличить LLM_MAX_TOKENS."
)


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


async def _save_user_message(
    session_factory: async_sessionmaker[AsyncSession],
    message: TelegramMessage,
) -> tuple[int, int]:
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
            return user.id, chat.id
        except Exception:
            await session.rollback()
            raise


async def _save_assistant_message(
    session_factory: async_sessionmaker[AsyncSession],
    chat_id: int,
    text: str,
    telegram_message_id: int | None,
) -> None:
    async with session_factory() as session:
        try:
            await save_message(
                session=session,
                chat_id=chat_id,
                user_id=None,
                role="assistant",
                text=text,
                message_type="text",
                telegram_message_id=telegram_message_id,
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
    settings: Settings,
) -> None:
    """Handle /status."""
    await _handle_with_db(
        session_factory,
        message,
        STATUS_TEXT.format(model=settings.llm_model),
        "Received /status command",
    )


@router.message(Command("llm_test"))
async def handle_llm_test(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    orchestrator: AgentOrchestrator,
) -> None:
    """Send a simple smoke-test prompt to the configured LLM."""
    logger.info("Received /llm_test command")
    chat_id: int | None = None
    try:
        _, chat_id = await _save_user_message(session_factory, message)
        response = await orchestrator.llm_client.generate_text(
            messages=[
                ChatMessage(
                    role="user",
                    content="Ответь одним коротким предложением: LLM работает.",
                )
            ],
            temperature=orchestrator.temperature,
            max_tokens=orchestrator.max_tokens,
        )
        answer_text = response.content.strip()
        if not answer_text:
            answer_text = LLM_EMPTY_RESPONSE_TEXT
    except LLMClientError:
        logger.exception("LLM test failed")
        answer_text = LLM_TEST_ERROR_TEXT
    except Exception:
        logger.exception("Failed to process /llm_test command")
        await message.answer(ERROR_TEXT)
        return

    response_message = await message.answer(answer_text)
    if chat_id is not None:
        try:
            await _save_assistant_message(
                session_factory,
                chat_id,
                answer_text,
                response_message.message_id,
            )
        except Exception:
            logger.exception("Failed to save /llm_test assistant message")


@router.message(F.text.func(lambda text: not text.startswith("/")))
async def handle_text(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    orchestrator: AgentOrchestrator,
) -> None:
    """Handle any non-command text message."""
    text_length = len(message.text or "")
    logger.info(
        "Received text message",
        extra={"telegram_chat_id": message.chat.id, "text_length": text_length},
    )
    try:
        _, chat_id = await _save_user_message(session_factory, message)
    except Exception:
        logger.exception("Failed to save user message")
        await message.answer(ERROR_TEXT)
        return

    try:
        async with session_factory() as session:
            recent_messages = await get_recent_messages(
                session=session,
                chat_id=chat_id,
                limit=orchestrator.max_context_messages,
            )
        answer_text = await orchestrator.generate_reply(recent_messages)
    except LLMClientError:
        logger.exception("LLM unavailable while processing text message")
        answer_text = LLM_ERROR_TEXT
    except Exception:
        logger.exception("Failed to generate Telegram reply")
        await message.answer(ERROR_TEXT)
        return

    response = await message.answer(answer_text)
    try:
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
        )
    except Exception:
        logger.exception("Failed to save assistant message")
