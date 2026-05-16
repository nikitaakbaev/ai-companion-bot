"""Telegram message handlers for the MVP."""

import logging
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message as TelegramMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentActionType, AgentDecision
from app.agent.tool_executor import ToolExecutionResult, ToolExecutor
from app.config import Settings
from app.database.models import Chat, User
from app.database.repositories import (
    get_or_create_chat,
    get_or_create_user,
    get_recent_messages,
    save_agent_action,
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
    "/status — проверить состояние\n"
    "/llm_test — проверить LLM\n"
    "/agent_test — проверить JSON agent loop"
)

STATUS_TEXT = (
    "Бот работает.\n\n"
    "Этап: 4\n"
    "Telegram: подключен\n"
    "База данных: подключена\n"
    "LLM: подключена через OpenAI-compatible API\n"
    "Модель: {model}\n"
    "Agent loop: включен\n"
    "Tool calling: JSON mode\n"
    "Реализованные tools: send_message, ignore\n"
    "Заглушки: remember, read_diary, sleep, take_photo, analyze_image\n"
    "Память/RAG: ещё не подключены\n"
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


async def _save_agent_action_record(
    session_factory: async_sessionmaker[AsyncSession],
    user_id: int | None,
    chat_id: int | None,
    decision: AgentDecision,
    result: ToolExecutionResult | None,
    status: str,
    error: str | None = None,
) -> None:
    async with session_factory() as session:
        await save_agent_action(
            session=session,
            user_id=user_id,
            chat_id=chat_id,
            action_type=decision.action.value,
            input_json=decision.model_dump(mode="json"),
            output_json=result.model_dump(mode="json") if result else None,
            status=status,
            error=error,
        )


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


@router.message(Command("agent_test"))
async def handle_agent_test(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    orchestrator: AgentOrchestrator,
) -> None:
    """Show a JSON agent decision without executing its tool."""
    logger.info("Received /agent_test command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message)
        async with session_factory() as session:
            recent_messages = await get_recent_messages(
                session=session,
                chat_id=chat_id,
                limit=orchestrator.max_context_messages,
            )
        decision = await orchestrator.decide(
            recent_messages=recent_messages,
            event_context={
                "event_type": "agent_test",
                "telegram_user_id": message.from_user.id if message.from_user else None,
                "username": message.from_user.username if message.from_user else None,
                "first_name": message.from_user.first_name if message.from_user else None,
                "text": "Пользователь проверяет JSON agent loop. Ответь коротко.",
            },
        )
        messages_text = "\n".join(f"- {text}" for text in decision.normalized_messages()) or "-"
        answer_text = (
            "Agent decision:\n\n"
            f"action: {decision.action.value}\n"
            f"emotion: {decision.emotion.value}\n"
            "messages:\n"
            f"{messages_text}"
        )
        response = await message.answer(answer_text)
        await _save_assistant_message(session_factory, chat_id, answer_text, response.message_id)
        await _save_agent_action_record(
            session_factory=session_factory,
            user_id=user_id,
            chat_id=chat_id,
            decision=decision,
            result=ToolExecutionResult(status="preview", output={"agent_test": True}),
            status="preview",
        )
    except LLMClientError:
        logger.exception("LLM unavailable while processing /agent_test")
        await message.answer(LLM_ERROR_TEXT)
    except Exception:
        logger.exception("Failed to process /agent_test command")
        await message.answer(ERROR_TEXT)


@router.message(F.text.func(lambda text: not text.startswith("/")))
async def handle_text(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    orchestrator: AgentOrchestrator,
    tool_executor: ToolExecutor,
) -> None:
    """Handle any non-command text message."""
    text_length = len(message.text or "")
    logger.info(
        "Received text message",
        extra={"telegram_chat_id": message.chat.id, "text_length": text_length},
    )
    try:
        user_id, chat_id = await _save_user_message(session_factory, message)
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
        decision = await orchestrator.decide(
            recent_messages=recent_messages,
            event_context={
                "event_type": "telegram_text_message",
                "telegram_user_id": message.from_user.id if message.from_user else None,
                "username": message.from_user.username if message.from_user else None,
                "first_name": message.from_user.first_name if message.from_user else None,
                "text": message.text,
            },
        )
    except LLMClientError:
        logger.exception("LLM unavailable while processing text message")
        response = await message.answer(LLM_ERROR_TEXT)
        await _save_assistant_message(session_factory, chat_id, LLM_ERROR_TEXT, response.message_id)
        return
    except Exception:
        logger.exception("Failed to generate Telegram reply")
        await message.answer(ERROR_TEXT)
        return

    result = await tool_executor.execute(decision=decision, telegram_chat_id=message.chat.id)
    logger.info(
        "Executed agent tool",
        extra={"action": decision.action.value, "tool_status": result.status},
    )
    await _save_agent_action_record(
        session_factory=session_factory,
        user_id=user_id,
        chat_id=chat_id,
        decision=decision,
        result=result,
        status=result.status,
        error=result.error,
    )

    if decision.action == AgentActionType.SEND_MESSAGE and result.status == "success":
        for sent_message in result.output.get("sent_messages", []):
            await _save_assistant_message(
                session_factory=session_factory,
                chat_id=chat_id,
                text=sent_message["text"],
                telegram_message_id=sent_message["message_id"],
            )
