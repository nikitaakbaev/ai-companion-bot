"""Application entrypoint."""

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError

from app.agent.orchestrator import AgentOrchestrator
from app.agent.response_verifier import AgentResponseVerifier
from app.agent.tool_executor import ToolExecutor
from app.bot.router import router
from app.config import get_settings
from app.database.session import create_engine_from_url, create_session_factory, init_db
from app.llm.openai_compatible import OpenAICompatibleLLMClient
from app.logging_config import setup_logging
from app.memory.diary import DiaryService
from app.memory.summarizer import DiarySummarizer

logger = logging.getLogger(__name__)


async def main() -> None:
    """Start the application."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Starting %s", settings.app_name)
    logger.info("Environment: %s", settings.app_env)
    logger.info("Debug: %s", settings.debug)
    logger.info("Database URL: %s", settings.database_url)
    logger.info("LLM base URL: %s", settings.llm_base_url)
    logger.info("LLM model: %s", settings.llm_model)
    logger.info("LLM backend: OpenAI-compatible API")

    engine = create_engine_from_url(settings.database_url)
    session_factory = create_session_factory(engine)
    await init_db(engine, settings.auto_create_tables)

    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN is not set. Telegram bot is not started.")
        await engine.dispose()
        return

    llm_client = OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
    )
    response_verifier = None
    if settings.response_verifier_enabled:
        verifier_model = settings.response_verifier_model or settings.llm_model
        verifier_client = OpenAICompatibleLLMClient(
            base_url=settings.response_verifier_base_url or settings.llm_base_url,
            api_key=settings.response_verifier_api_key or settings.llm_api_key,
            model=verifier_model,
            timeout_seconds=settings.llm_timeout_seconds,
            default_temperature=0,
            default_max_tokens=settings.response_verifier_max_tokens,
        )
        response_verifier = AgentResponseVerifier(
            llm_client=verifier_client,
            max_context_messages=settings.agent_context_messages,
            max_tokens=settings.response_verifier_max_tokens,
        )
        logger.info("Response verifier enabled with model: %s", verifier_model)

    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        max_context_messages=settings.agent_context_messages,
        temperature=settings.agent_temperature,
        max_tokens=settings.agent_max_tokens,
        response_verifier=response_verifier,
    )
    diary_model = settings.diary_reflection_model or settings.llm_model
    diary_llm_client = (
        llm_client
        if diary_model == settings.llm_model
        else OpenAICompatibleLLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=diary_model,
            timeout_seconds=settings.llm_timeout_seconds,
            default_temperature=0.2,
            default_max_tokens=settings.llm_max_tokens,
        )
    )
    diary_summarizer = DiarySummarizer(
        llm_client=diary_llm_client,
        max_entries_per_run=settings.diary_max_entries_per_run,
        max_input_chars=settings.diary_max_input_chars,
    )
    diary_service = DiaryService(
        summarizer=diary_summarizer,
        min_messages=settings.diary_min_messages,
        max_messages=settings.diary_max_messages,
        lookback_hours=settings.diary_lookback_hours,
        skip_if_exists_for_date=settings.diary_skip_if_exists_for_date,
    )

    bot = Bot(token=settings.telegram_bot_token, session=AiohttpSession(timeout=120))
    tool_executor = ToolExecutor(
        bot=bot,
        diary_service=diary_service,
        max_delay_seconds=settings.agent_max_delay_seconds,
        typing_seconds=settings.agent_typing_seconds,
    )
    dp = Dispatcher()
    dp.include_router(router)
    dp["session_factory"] = session_factory
    dp["orchestrator"] = orchestrator
    dp["tool_executor"] = tool_executor
    dp["diary_service"] = diary_service
    dp["settings"] = settings

    logger.info("Starting Telegram polling")
    try:
        await dp.start_polling(bot)
    except TelegramNetworkError:
        logger.exception(
            "Cannot connect to Telegram API. Check network, VPN/proxy, firewall, or try again later."
        )
    finally:
        await bot.session.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
