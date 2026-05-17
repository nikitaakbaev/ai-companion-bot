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
from app.bot.formatters import (
    format_actions,
    format_diary,
    format_diary_full,
    format_history,
    format_settings,
    format_sleep_result,
)
from app.config import Settings
from app.database.models import AgentState, BotSettings, Chat, User
from app.database.repositories import (
    get_bot_settings,
    get_or_create_chat,
    get_or_create_agent_state,
    get_or_create_bot_settings,
    get_or_create_user,
    get_recent_agent_actions,
    get_recent_diary_entries,
    get_recent_messages,
    now_for_agent_state,
    save_agent_action,
    save_message,
    update_agent_state,
)
from app.llm.client import ChatMessage, LLMClientError
from app.memory.diary import DiaryService

logger = logging.getLogger(__name__)
router = Router()

START_TEXT = (
    "Привет. Я AI companion bot.\n\n"
    "Сейчас уже работает Telegram, база, LLM и JSON agent loop.\n"
    "На следующем этапе появится дневник: я смогу сжимать историю переписки в воспоминания."
)

HELP_TEXT = (
    "Доступные команды:\n\n"
    "/start — запустить бота\n"
    "/help — показать помощь\n"
    "/status — проверить состояние\n"
    "/llm_test — проверить LLM\n"
    "/agent_test — проверить JSON agent loop\n"
    "/settings — показать настройки\n"
    "/history — последние сообщения\n"
    "/actions — последние действия агента\n"
    "/sleep — сжать историю переписки в дневник\n"
    "/diary — кратко показать дневник\n"
    "/diary_full — показать полные последние записи"
)

STATUS_TEXT = (
    "Бот работает.\n\n"
    "Этап: 6\n"
    "Telegram: подключен\n"
    "База данных: подключена\n"
    "LLM: подключена через OpenAI-compatible API\n"
    "Модель: {model}\n"
    "Agent loop: включен\n"
    "Tool calling: JSON mode\n"
    "Дневник: {diary_status}\n"
    "RAG: ещё не подключен\n"
    "Vision: ещё не подключен\n"
    "Проактивность: ещё не подключена"
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
        language_code=message.from_user.language_code,
        is_bot=message.from_user.is_bot,
    )
    chat = await get_or_create_chat(
        session=session,
        telegram_chat_id=message.chat.id,
        user_id=user.id,
        title=message.chat.title,
        chat_type=str(message.chat.type),
    )
    return user, chat


async def _ensure_user_defaults(
    session: AsyncSession,
    user_id: int,
    settings: Settings,
) -> tuple[BotSettings, AgentState]:
    bot_settings = await get_or_create_bot_settings(session, user_id, settings)
    agent_state = await get_or_create_agent_state(session, user_id)
    return bot_settings, agent_state


async def _save_interaction(
    session_factory: async_sessionmaker[AsyncSession],
    message: TelegramMessage,
    response_text: str,
    settings: Settings,
) -> None:
    async with session_factory() as session:
        try:
            user, chat = await _get_or_create_context(session, message)
            await _ensure_user_defaults(session, user.id, settings)
            await save_message(
                session=session,
                chat_id=chat.id,
                user_id=user.id,
                role="user",
                text=message.text,
                message_type="text",
                telegram_message_id=message.message_id,
                max_stored_message_length=settings.max_stored_message_length,
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
                max_stored_message_length=settings.max_stored_message_length,
            )
        except Exception:
            await session.rollback()
            raise


async def _save_user_message(
    session_factory: async_sessionmaker[AsyncSession],
    message: TelegramMessage,
    settings: Settings,
) -> tuple[int, int]:
    async with session_factory() as session:
        try:
            user, chat = await _get_or_create_context(session, message)
            await _ensure_user_defaults(session, user.id, settings)
            await save_message(
                session=session,
                chat_id=chat.id,
                user_id=user.id,
                role="user",
                text=message.text,
                message_type="text",
                telegram_message_id=message.message_id,
                reply_to_message_id=message.reply_to_message.message_id
                if message.reply_to_message
                else None,
                max_stored_message_length=settings.max_stored_message_length,
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
    settings: Settings,
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
                max_stored_message_length=settings.max_stored_message_length,
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
    settings: Settings,
    log_context: dict[str, Any] | None = None,
) -> None:
    logger.info(log_message, extra=log_context or {})
    try:
        await _save_interaction(session_factory, message, response_text, settings)
    except Exception:
        logger.exception("Failed to process Telegram message")
        await message.answer(ERROR_TEXT)


@router.message(Command("start"))
async def handle_start(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Handle /start."""
    await _handle_with_db(session_factory, message, START_TEXT, "Received /start command", settings)


@router.message(Command("help"))
async def handle_help(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Handle /help."""
    await _handle_with_db(session_factory, message, HELP_TEXT, "Received /help command", settings)


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
        STATUS_TEXT.format(
            model=settings.llm_model,
            diary_status="включен" if settings.diary_enabled else "выключен в настройках",
        ),
        "Received /status command",
        settings,
    )


@router.message(Command("llm_test"))
async def handle_llm_test(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    orchestrator: AgentOrchestrator,
    settings: Settings,
) -> None:
    """Send a simple smoke-test prompt to the configured LLM."""
    logger.info("Received /llm_test command")
    chat_id: int | None = None
    try:
        _, chat_id = await _save_user_message(session_factory, message, settings)
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
                settings,
            )
        except Exception:
            logger.exception("Failed to save /llm_test assistant message")


@router.message(Command("agent_test"))
async def handle_agent_test(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    orchestrator: AgentOrchestrator,
    settings: Settings,
) -> None:
    """Show a JSON agent decision without executing its tool."""
    logger.info("Received /agent_test command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        async with session_factory() as session:
            recent_messages = await get_recent_messages(
                session=session,
                chat_id=chat_id,
                limit=orchestrator.max_context_messages,
            )
            bot_settings = await get_bot_settings(session, user_id)
            agent_state = await get_or_create_agent_state(session, user_id)
        decision = await orchestrator.decide(
            recent_messages=recent_messages,
            event_context={
                "event_type": "agent_test",
                "telegram_user_id": message.from_user.id if message.from_user else None,
                "username": message.from_user.username if message.from_user else None,
                "first_name": message.from_user.first_name if message.from_user else None,
                "text": "Пользователь проверяет JSON agent loop. Ответь коротко.",
            },
            bot_settings=bot_settings,
            agent_state=agent_state,
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
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
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


@router.message(Command("settings"))
async def handle_settings(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Show per-user bot settings."""
    logger.info("Received /settings command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        async with session_factory() as session:
            bot_settings = await get_or_create_bot_settings(session, user_id, settings)
        answer_text = format_settings(bot_settings)
        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
    except Exception:
        logger.exception("Failed to process /settings command")
        await message.answer(ERROR_TEXT)


@router.message(Command("history"))
async def handle_history(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Show recent stored messages."""
    logger.info("Received /history command")
    try:
        _, chat_id = await _save_user_message(session_factory, message, settings)
        async with session_factory() as session:
            messages = await get_recent_messages(session, chat_id=chat_id, limit=10)
        answer_text = format_history(messages)
        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
    except Exception:
        logger.exception("Failed to process /history command")
        await message.answer(ERROR_TEXT)


@router.message(Command("actions"))
async def handle_actions(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Show recent agent actions."""
    logger.info("Received /actions command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        async with session_factory() as session:
            actions = await get_recent_agent_actions(session, user_id=user_id, limit=10)
        answer_text = format_actions(actions)
        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
    except Exception:
        logger.exception("Failed to process /actions command")
        await message.answer(ERROR_TEXT)


@router.message(Command("diary"))
async def handle_diary(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Show recent diary entries."""
    logger.info("Received /diary command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        async with session_factory() as session:
            entries = await get_recent_diary_entries(session, user_id=user_id, limit=10)
        answer_text = format_diary(entries)
        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
    except Exception:
        logger.exception("Failed to process /diary command")
        await message.answer(ERROR_TEXT)


@router.message(Command("diary_full"))
async def handle_diary_full(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Show recent diary entries in full form."""
    logger.info("Received /diary_full command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        async with session_factory() as session:
            entries = await get_recent_diary_entries(session, user_id=user_id, limit=3)
        answer_text = format_diary_full(entries)
        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
    except Exception:
        logger.exception("Failed to process /diary_full command")
        await message.answer(ERROR_TEXT)


@router.message(Command("sleep"))
async def handle_sleep(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    diary_service: DiaryService,
    settings: Settings,
) -> None:
    """Create diary entries from recent conversation history."""
    logger.info("Received /sleep command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        if not settings.diary_enabled:
            answer_text = "Дневник выключен в настройках."
            result = None
        else:
            async with session_factory() as session:
                result = await diary_service.create_daily_summary(session=session, user_id=user_id)
            answer_text = format_sleep_result(result)

        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
        await _save_agent_action_record(
            session_factory=session_factory,
            user_id=user_id,
            chat_id=chat_id,
            decision=AgentDecision(action=AgentActionType.SLEEP, messages=[]),
            result=ToolExecutionResult(
                status=result.status if result else "disabled",
                output=result.model_dump(mode="json") if result else {"disabled": True},
            ),
            status=result.status if result else "disabled",
        )
    except Exception:
        logger.exception("Failed to process /sleep command")
        await message.answer(ERROR_TEXT)


@router.message(F.text.func(lambda text: not text.startswith("/")))
async def handle_text(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    orchestrator: AgentOrchestrator,
    tool_executor: ToolExecutor,
    settings: Settings,
) -> None:
    """Handle any non-command text message."""
    text_length = len(message.text or "")
    logger.info(
        "Received text message",
        extra={"telegram_chat_id": message.chat.id, "text_length": text_length},
    )
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
    except Exception:
        logger.exception("Failed to save user message")
        await message.answer(ERROR_TEXT)
        return

    try:
        async with session_factory() as session:
            recent_messages = await get_recent_messages(
                session=session,
                chat_id=chat_id,
                limit=settings.message_history_limit,
            )
            bot_settings = await get_bot_settings(session, user_id)
            agent_state = await get_or_create_agent_state(session, user_id)
        event_context = {
            "event_type": "telegram_text_message",
            "telegram_user_id": message.from_user.id if message.from_user else None,
            "username": message.from_user.username if message.from_user else None,
            "first_name": message.from_user.first_name if message.from_user else None,
            "text": message.text,
        }
        if settings.agent_plain_text_mode:
            decision = await orchestrator.decide_plain_reply(
                recent_messages=recent_messages,
                event_context=event_context,
                bot_settings=bot_settings,
                agent_state=agent_state,
            )
        else:
            decision = await orchestrator.decide(
                recent_messages=recent_messages,
                event_context=event_context,
                bot_settings=bot_settings,
                agent_state=agent_state,
            )
    except LLMClientError:
        logger.exception("LLM unavailable while processing text message")
        response = await message.answer(LLM_ERROR_TEXT)
        await _save_assistant_message(
            session_factory,
            chat_id,
            LLM_ERROR_TEXT,
            response.message_id,
            settings,
        )
        return
    except Exception:
        logger.exception("Failed to generate Telegram reply")
        await message.answer(ERROR_TEXT)
        return

    async with session_factory() as session:
        result = await tool_executor.execute(
            decision=decision,
            telegram_chat_id=message.chat.id,
            session=session,
            user_id=user_id,
        )
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
    async with session_factory() as session:
        await update_agent_state(
            session=session,
            user_id=user_id,
            last_emotion=decision.emotion.value,
            last_action_type=decision.action.value,
            last_interaction_at=now_for_agent_state(),
        )

    if decision.action == AgentActionType.SEND_MESSAGE and result.status == "success":
        for sent_message in result.output.get("sent_messages", []):
            await _save_assistant_message(
                session_factory=session_factory,
                chat_id=chat_id,
                text=sent_message["text"],
                telegram_message_id=sent_message["message_id"],
                settings=settings,
            )
