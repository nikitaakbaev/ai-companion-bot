"""Telegram message handlers for the MVP."""

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    format_memory,
    format_memory_search,
    format_settings,
    format_sleep_result,
)
from app.config import Settings
from app.database.models import AgentState, BotSettings, Chat, User
from app.database.repositories import (
    get_bot_settings,
    create_media_file,
    delete_diary_entries_for_user,
    get_or_create_chat,
    get_or_create_agent_state,
    get_or_create_bot_settings,
    get_or_create_user,
    get_diary_entries_for_user,
    get_recent_agent_actions,
    get_recent_diary_entries,
    get_recent_messages,
    now_for_agent_state,
    save_agent_action,
    save_message,
    update_agent_state,
)
from app.llm.client import ChatMessage, LLMClientError
from app.llm.embeddings import EmbeddingError
from app.llm.vision import VisionClient
from app.image_generation.stable_waifu_provider import STABLE_WAIFU_MODELS, STABLE_WAIFU_PRESETS
from app.image_generation.telegram_userbot import TelegramUserbotClient
from app.memory.diary import DiaryService
from app.memory.profiles import profiles_from_settings
from app.memory.rag import MemoryIndexer, MemoryRetriever, RelevantMemory
from app.memory.vector_store import VectorStoreError

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

HELP_TEXT += (
    "\n/memory - show memory and embedding status"
    "\n/memory_search <query> - search memories"
    "\n/memory_reindex - rebuild diary embeddings"
    "\n/memory_reset - delete diary memory and embeddings"
    "\n/photo_test <description> - generate a ComfyUI photo"
    "\n/stable_waifu_test - generate a Stable Waifu test image"
    "\n/stable_waifu_settings - show Stable Waifu settings"
    "\n/stable_waifu_models - show supported Stable Waifu models"
    "\n/stable_waifu_login_status - show Telethon userbot status"
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
STATUS_TEXT = (
    "Bot is running.\n\n"
    "Stage: 7\n"
    "Telegram: connected\n"
    "Database: connected\n"
    "LLM: OpenAI-compatible API\n"
    "Model: {model}\n"
    "Agent loop: enabled\n"
    "Tool calling: JSON mode\n"
    "Diary: {diary_status}\n"
    "RAG memory: enabled\n"
    "Vector store: chroma\n"
    "Embeddings: configured\n"
    "Vision: not connected yet\n"
    "Image generation: not connected yet\n"
    "Proactivity: not connected yet"
)

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


async def _save_photo_message(
    session_factory: async_sessionmaker[AsyncSession],
    message: TelegramMessage,
    settings: Settings,
) -> tuple[int, int, dict[str, Any]]:
    """Download a Telegram photo and save it with its caption for agent context."""
    if not message.photo:
        raise ValueError("Telegram message has no photo")

    photo = max(message.photo, key=lambda item: item.file_size or item.width * item.height)
    media_dir = Path(settings.media_storage_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    local_path = media_dir / f"{uuid4().hex}.jpg"

    bot_file = await message.bot.get_file(photo.file_id)
    if bot_file.file_path is None:
        raise ValueError("Telegram photo file path is missing")
    await message.bot.download_file(bot_file.file_path, destination=str(local_path))

    caption = (message.caption or "").strip()
    stored_text = _photo_context_text(caption)
    async with session_factory() as session:
        try:
            user, chat = await _get_or_create_context(session, message)
            await _ensure_user_defaults(session, user.id, settings)
            media_file = await create_media_file(
                session=session,
                user_id=user.id,
                chat_id=chat.id,
                file_type="photo",
                telegram_file_id=photo.file_id,
                telegram_file_unique_id=photo.file_unique_id,
                local_path=str(local_path),
                metadata_json={
                    "width": photo.width,
                    "height": photo.height,
                    "file_size": photo.file_size,
                    "caption": caption or None,
                },
            )
            await save_message(
                session=session,
                chat_id=chat.id,
                user_id=user.id,
                role="user",
                text=stored_text,
                message_type="image",
                telegram_message_id=message.message_id,
                media_file_id=media_file.id,
                metadata_json={
                    "caption": caption or None,
                    "local_path": str(local_path),
                    "telegram_file_id": photo.file_id,
                },
                max_stored_message_length=settings.max_stored_message_length,
            )
            return user.id, chat.id, {
                "media_file_id": media_file.id,
                "local_path": str(local_path),
                "caption": caption,
                "telegram_file_id": photo.file_id,
                "width": photo.width,
                "height": photo.height,
            }
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


async def _save_assistant_photo_message(
    session_factory: async_sessionmaker[AsyncSession],
    chat_id: int,
    user_id: int | None,
    caption: str | None,
    telegram_message_id: int | None,
    image_path: str,
    metadata: dict[str, Any],
    settings: Settings,
) -> None:
    async with session_factory() as session:
        try:
            media_file = await create_media_file(
                session=session,
                user_id=user_id,
                chat_id=chat_id,
                file_type="generated_photo",
                local_path=image_path,
                metadata_json=metadata,
            )
            await save_message(
                session=session,
                chat_id=chat_id,
                user_id=None,
                role="assistant",
                text=caption,
                message_type="image",
                telegram_message_id=telegram_message_id,
                media_file_id=media_file.id,
                metadata_json=metadata,
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
    input_extra: dict[str, Any] | None = None,
) -> None:
    async with session_factory() as session:
        input_json = decision.model_dump(mode="json")
        if input_extra:
            input_json.update(input_extra)
        await save_agent_action(
            session=session,
            user_id=user_id,
            chat_id=chat_id,
            action_type=decision.action.value,
            input_json=input_json,
            output_json=result.model_dump(mode="json") if result else None,
            status=status,
            error=error,
        )


async def _maybe_create_auto_memory(
    session_factory: async_sessionmaker[AsyncSession],
    diary_service: DiaryService,
    user_id: int,
    settings: Settings,
) -> None:
    if not settings.diary_enabled or not settings.diary_auto_create_enabled:
        return

    try:
        async with session_factory() as session:
            result = await diary_service.create_daily_summary(session=session, user_id=user_id)
        if result.status == "created":
            logger.info(
                "Automatic diary memory created",
                extra={
                    "created_count": result.created_count,
                    "indexed_count": result.indexed_count,
                },
            )
    except Exception:
        logger.exception("Failed to create automatic diary memory")


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


def _command_args(text: str | None) -> str:
    """Return command arguments after the first whitespace."""
    if not text:
        return ""
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _photo_context_text(caption: str) -> str:
    return f"[Photo attached]\nCaption: {caption}" if caption else "[Photo attached]"


def _profiles_context(settings: Settings) -> dict[str, dict[str, str]]:
    user_profile, character_profile = profiles_from_settings(settings)
    return {"user": user_profile, "character": character_profile}


def _image_generation_context(settings: Settings) -> dict[str, str | int | bool]:
    return {
        "provider": settings.image_generation_provider,
        "stable_waifu_model": settings.stable_waifu_model,
        "stable_waifu_preset": settings.stable_waifu_preset,
        "orientation": settings.stable_waifu_orientation,
        "aspect_ratio": settings.stable_waifu_aspect_ratio,
        "scene_tags_format": "lowercase comma-separated anime tags only",
        "base_tags_injected": True,
        "max_scene_tags": 30,
        "nsfw_level": settings.stable_waifu_nsfw_level,
    }


def _photo_request_detected(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.casefold()
    photo_words = ("фото", "фотку", "фоточку", "селфи", "картинку", "снимок", "photo", "selfie", "picture")
    request_words = (
        "отправ",
        "пришли",
        "скинь",
        "покаж",
        "сделай",
        "попробуй",
        "можешь",
        "send",
        "show",
        "take",
        "try",
    )
    return any(word in lowered for word in photo_words) and any(
        word in lowered for word in request_words
    )


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


@router.message(Command("memory"))
async def handle_memory(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Show diary memory with embedding status."""
    logger.info("Received /memory command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        async with session_factory() as session:
            entries = await get_recent_diary_entries(session, user_id=user_id, limit=10)
        answer_text = format_memory(entries)
        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
    except Exception:
        logger.exception("Failed to process /memory command")
        await message.answer(ERROR_TEXT)


@router.message(Command("memory_search"))
async def handle_memory_search(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    memory_retriever: MemoryRetriever | None,
    settings: Settings,
) -> None:
    """Search long-term memory."""
    logger.info("Received /memory_search command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        query = _command_args(message.text)
        if not query:
            answer_text = "Usage: /memory_search <query>"
        elif not settings.rag_enabled or memory_retriever is None:
            answer_text = "RAG memory is disabled."
        else:
            try:
                memories = await memory_retriever.retrieve(query=query, user_id=user_id)
            except (EmbeddingError, VectorStoreError) as exc:
                logger.warning("RAG memory is unavailable: %s", exc)
                answer_text = f"RAG memory is unavailable: {exc}"
            else:
                answer_text = format_memory_search(memories)
        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
    except Exception:
        logger.exception("Failed to process /memory_search command")
        await message.answer(ERROR_TEXT)


@router.message(Command("memory_reindex"))
async def handle_memory_reindex(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    memory_indexer: MemoryIndexer | None,
    settings: Settings,
) -> None:
    """Rebuild diary embeddings for the current user."""
    logger.info("Received /memory_reindex command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        if not settings.rag_enabled or memory_indexer is None:
            answer_text = "RAG memory is disabled."
        else:
            async with session_factory() as session:
                entries = await get_diary_entries_for_user(session, user_id=user_id)
                if not entries:
                    indexed = 0
                else:
                    try:
                        indexed = await memory_indexer.reindex_user_diary(session, user_id=user_id)
                    except (EmbeddingError, VectorStoreError) as exc:
                        logger.warning("RAG memory is unavailable: %s", exc)
                        answer_text = f"RAG memory is unavailable: {exc}"
                    else:
                        answer_text = f"Reindex complete.\n\nIndexed entries: {indexed}"
            if not entries:
                answer_text = "Diary is empty. Nothing to index."
        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
    except Exception:
        logger.exception("Failed to process /memory_reindex command")
        await message.answer(ERROR_TEXT)


@router.message(Command("memory_reset"))
async def handle_memory_reset(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    memory_indexer: MemoryIndexer | None,
    settings: Settings,
) -> None:
    """Delete diary memory and vector embeddings for the current user."""
    logger.info("Received /memory_reset command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        async with session_factory() as session:
            if memory_indexer is not None:
                deleted = await memory_indexer.reset_user_memory(session, user_id=user_id)
            else:
                deleted = await delete_diary_entries_for_user(session, user_id=user_id)
        answer_text = f"Memory reset complete.\n\nDeleted diary entries: {deleted}"
        response = await message.answer(answer_text)
        await _save_assistant_message(
            session_factory,
            chat_id,
            answer_text,
            response.message_id,
            settings,
        )
    except Exception:
        logger.exception("Failed to process /memory_reset command")
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


@router.message(Command("photo_test"))
async def handle_photo_test(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    tool_executor: ToolExecutor,
    settings: Settings,
) -> None:
    """Generate a test photo through ComfyUI."""
    logger.info("Received /photo_test command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        description = _command_args(message.text) or "telegram selfie, looking at camera"
        decision = AgentDecision(
            action=AgentActionType.TAKE_PHOTO,
            messages=[],
            tool_input={
                "description": description,
                "mood": "playful",
                "style": "anime selfie",
            },
        )
        async with session_factory() as session:
            result = await tool_executor.execute(
                decision=decision,
                telegram_chat_id=message.chat.id,
                session=session,
                user_id=user_id,
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
        if result.status != "success":
            await message.answer(f"Image generation failed: {result.error or result.status}")
    except Exception:
        logger.exception("Failed to process /photo_test command")
        await message.answer(ERROR_TEXT)


@router.message(Command("stable_waifu_settings"))
async def handle_stable_waifu_settings(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Show Stable Waifu generation settings."""
    logger.info("Received /stable_waifu_settings command")
    answer_text = (
        "Stable Waifu settings:\n\n"
        f"Provider: {settings.image_generation_provider}\n"
        f"Enabled: {settings.stable_waifu_enabled}\n"
        f"Bot: @{settings.stable_waifu_bot_username.lstrip('@')}\n"
        f"Model: {settings.stable_waifu_model}\n"
        f"Preset: {settings.stable_waifu_preset}\n"
        f"Orientation: {settings.stable_waifu_orientation}\n"
        f"Aspect ratio: {settings.stable_waifu_aspect_ratio}\n"
        f"Model page switches: {settings.stable_waifu_model_search_max_page_switches}\n"
        f"Timeout: {settings.stable_waifu_timeout_seconds}s\n"
        f"Poll interval: {settings.stable_waifu_poll_interval_seconds}s"
    )
    await _handle_with_db(
        session_factory,
        message,
        answer_text,
        "Received /stable_waifu_settings command",
        settings,
    )


@router.message(Command("stable_waifu_models"))
async def handle_stable_waifu_models(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Show supported Stable Waifu models and presets."""
    logger.info("Received /stable_waifu_models command")
    answer_text = (
        "Stable Waifu models:\n"
        + "\n".join(f"- {model}" for model in STABLE_WAIFU_MODELS)
        + "\n\nPresets:\n"
        + "\n".join(f"- {preset}" for preset in STABLE_WAIFU_PRESETS)
    )
    await _handle_with_db(
        session_factory,
        message,
        answer_text,
        "Received /stable_waifu_models command",
        settings,
    )


@router.message(Command("stable_waifu_login_status"))
async def handle_stable_waifu_login_status(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    telegram_userbot: TelegramUserbotClient | None,
    settings: Settings,
) -> None:
    """Show Telethon userbot status."""
    logger.info("Received /stable_waifu_login_status command")
    answer_text = (
        "Stable Waifu userbot:\n\n"
        f"Configured: {settings.telegram_userbot_api_id is not None and bool(settings.telegram_userbot_api_hash)}\n"
        f"Started: {telegram_userbot.is_started if telegram_userbot is not None else False}\n"
        f"Session: {settings.telegram_userbot_session_path}"
    )
    await _handle_with_db(
        session_factory,
        message,
        answer_text,
        "Received /stable_waifu_login_status command",
        settings,
    )


@router.message(Command("stable_waifu_test"))
async def handle_stable_waifu_test(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    tool_executor: ToolExecutor,
    settings: Settings,
) -> None:
    """Generate a test image through the selected Stable Waifu provider."""
    logger.info("Received /stable_waifu_test command")
    try:
        user_id, chat_id = await _save_user_message(session_factory, message, settings)
        if settings.image_generation_provider.strip().lower() != "stable_waifu_telegram":
            await message.answer("Stable Waifu provider is not selected.")
            return

        decision = AgentDecision(
            action=AgentActionType.TAKE_PHOTO,
            messages=["Stable Waifu test image."],
            tool_input={
                "description": "1girl, blue hair, black hoodie, selfie, cozy room, night",
                "mood": "",
                "style": "anime, best quality",
            },
        )
        async with session_factory() as session:
            result = await tool_executor.execute(
                decision=decision,
                telegram_chat_id=message.chat.id,
                session=session,
                user_id=user_id,
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
        if result.status != "success":
            await message.answer(f"Stable Waifu generation failed: {result.error or result.status}")
        else:
            await _save_assistant_photo_message(
                session_factory=session_factory,
                chat_id=chat_id,
                user_id=user_id,
                caption=str(result.output.get("caption") or "") or None,
                telegram_message_id=result.output.get("photo_message_id"),
                image_path=str(result.output.get("image_path") or ""),
                metadata={
                    "provider": result.output.get("provider"),
                    "prompt": result.output.get("prompt"),
                    "negative_prompt": result.output.get("negative_prompt"),
                    "model": result.output.get("model"),
                    "preset": result.output.get("preset"),
                    "telegram_message_id": result.output.get("telegram_message_id"),
                    "metadata": result.output.get("metadata") or {},
                },
                settings=settings,
            )
    except Exception:
        logger.exception("Failed to process /stable_waifu_test command")
        await message.answer(ERROR_TEXT)


@router.message(F.photo)
async def handle_photo(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    orchestrator: AgentOrchestrator,
    tool_executor: ToolExecutor,
    memory_retriever: MemoryRetriever | None,
    vision_client: VisionClient | None,
    diary_service: DiaryService,
    settings: Settings,
) -> None:
    """Handle incoming photos with optional captions."""
    caption = (message.caption or "").strip()
    logger.info(
        "Received photo message",
        extra={"telegram_chat_id": message.chat.id, "caption_length": len(caption)},
    )
    try:
        user_id, chat_id, photo_context = await _save_photo_message(
            session_factory,
            message,
            settings,
        )
    except Exception:
        logger.exception("Failed to save photo message")
        await message.answer(ERROR_TEXT)
        return

    image_analysis: str | None = None
    if settings.vision_enabled and vision_client is not None:
        try:
            image_analysis = await vision_client.describe_image(
                photo_context["local_path"],
                prompt=_vision_prompt(caption, settings),
            )
        except LLMClientError as exc:
            logger.warning("Vision analysis unavailable; continuing with caption only: %s", exc)
        except Exception:
            logger.exception("Failed to analyze photo; continuing with caption only")

    relevant_memories: list[RelevantMemory] = []
    try:
        async with session_factory() as session:
            recent_messages = await get_recent_messages(
                session=session,
                chat_id=chat_id,
                limit=settings.message_history_limit,
            )
            bot_settings = await get_bot_settings(session, user_id)
            agent_state = await get_or_create_agent_state(session, user_id)
        if settings.rag_enabled and memory_retriever is not None and caption:
            try:
                relevant_memories = await memory_retriever.retrieve(
                    query=caption,
                    user_id=user_id,
                )
            except (EmbeddingError, VectorStoreError) as exc:
                logger.warning("RAG memory is unavailable; continuing without memories: %s", exc)
            except Exception:
                logger.exception("Failed to retrieve relevant memories")
        decision = await orchestrator.decide(
            recent_messages=recent_messages,
            event_context={
                "event_type": "telegram_photo_message",
                "telegram_user_id": message.from_user.id if message.from_user else None,
                "username": message.from_user.username if message.from_user else None,
                "first_name": message.from_user.first_name if message.from_user else None,
                "text": _photo_context_text(caption),
                "caption": caption,
                "photo_request_detected": _photo_request_detected(caption),
                "photo": photo_context,
                "image_analysis_available": bool(image_analysis),
                "image_analysis": image_analysis,
                "profiles": _profiles_context(settings),
                "image_generation": _image_generation_context(settings),
            },
            bot_settings=bot_settings,
            agent_state=agent_state,
            relevant_memories=relevant_memories,
        )
    except LLMClientError:
        logger.exception("LLM unavailable while processing photo message")
        await message.answer(LLM_ERROR_TEXT)
        return
    except Exception:
        logger.exception("Failed to generate Telegram photo reply")
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
        input_extra={
            "memory": {
                "count": len(relevant_memories),
                "scores": [memory.score for memory in relevant_memories],
                "diary_entry_ids": [memory.diary_entry_id for memory in relevant_memories],
            },
            "photo": photo_context,
            "image_analysis": image_analysis,
        },
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
    if decision.action == AgentActionType.TAKE_PHOTO and result.status == "success":
        caption = str(result.output.get("caption") or "").strip()
        await _save_assistant_photo_message(
            session_factory=session_factory,
            chat_id=chat_id,
            user_id=user_id,
            caption=caption or None,
            telegram_message_id=result.output.get("photo_message_id"),
            image_path=str(result.output.get("image_path") or ""),
            metadata={
                "provider": result.output.get("provider"),
                "prompt": result.output.get("prompt"),
                "negative_prompt": result.output.get("negative_prompt"),
                "model": result.output.get("model"),
                "preset": result.output.get("preset"),
                "telegram_message_id": result.output.get("telegram_message_id"),
                "metadata": result.output.get("metadata") or {},
            },
            settings=settings,
        )
    await _maybe_create_auto_memory(
        session_factory=session_factory,
        diary_service=diary_service,
        user_id=user_id,
        settings=settings,
    )


def _vision_prompt(caption: str, settings: Settings) -> str:
    if caption:
        return f"{settings.vision_prompt}\n\nUser caption/context: {caption}"
    return settings.vision_prompt


@router.message(F.text.func(lambda text: bool(text) and not text.startswith("/")))
async def handle_text(
    message: TelegramMessage,
    session_factory: async_sessionmaker[AsyncSession],
    orchestrator: AgentOrchestrator,
    tool_executor: ToolExecutor,
    memory_retriever: MemoryRetriever | None,
    diary_service: DiaryService,
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

    relevant_memories: list[RelevantMemory] = []
    try:
        async with session_factory() as session:
            recent_messages = await get_recent_messages(
                session=session,
                chat_id=chat_id,
                limit=settings.message_history_limit,
            )
            bot_settings = await get_bot_settings(session, user_id)
            agent_state = await get_or_create_agent_state(session, user_id)
        if settings.rag_enabled and memory_retriever is not None and message.text:
            try:
                relevant_memories = await memory_retriever.retrieve(
                    query=message.text,
                    user_id=user_id,
                )
            except (EmbeddingError, VectorStoreError) as exc:
                logger.warning("RAG memory is unavailable; continuing without memories: %s", exc)
            except Exception:
                logger.exception("Failed to retrieve relevant memories")
        decision = await orchestrator.decide(
            recent_messages=recent_messages,
            event_context={
                "event_type": "telegram_text_message",
                "telegram_user_id": message.from_user.id if message.from_user else None,
                "username": message.from_user.username if message.from_user else None,
                "first_name": message.from_user.first_name if message.from_user else None,
                "text": message.text,
                "photo_request_detected": _photo_request_detected(message.text),
                "profiles": _profiles_context(settings),
                "image_generation": _image_generation_context(settings),
            },
            bot_settings=bot_settings,
            agent_state=agent_state,
            relevant_memories=relevant_memories,
        )
    except LLMClientError:
        logger.exception("LLM unavailable while processing text message")
        await message.answer(LLM_ERROR_TEXT)
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
        input_extra={
            "memory": {
                "count": len(relevant_memories),
                "scores": [memory.score for memory in relevant_memories],
                "diary_entry_ids": [memory.diary_entry_id for memory in relevant_memories],
            }
        },
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
    if decision.action == AgentActionType.TAKE_PHOTO and result.status == "success":
        caption = str(result.output.get("caption") or "").strip()
        await _save_assistant_photo_message(
            session_factory=session_factory,
            chat_id=chat_id,
            user_id=user_id,
            caption=caption or None,
            telegram_message_id=result.output.get("photo_message_id"),
            image_path=str(result.output.get("image_path") or ""),
            metadata={
                "provider": result.output.get("provider"),
                "prompt": result.output.get("prompt"),
                "negative_prompt": result.output.get("negative_prompt"),
                "model": result.output.get("model"),
                "preset": result.output.get("preset"),
                "telegram_message_id": result.output.get("telegram_message_id"),
                "metadata": result.output.get("metadata") or {},
            },
            settings=settings,
        )
    await _maybe_create_auto_memory(
        session_factory=session_factory,
        diary_service=diary_service,
        user_id=user_id,
        settings=settings,
    )
