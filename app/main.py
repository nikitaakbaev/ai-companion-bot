"""Application entrypoint."""

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError

from app.agent.orchestrator import AgentOrchestrator
from app.agent.tool_executor import ToolExecutor
from app.bot.router import router
from app.config import get_settings
from app.database.session import create_engine_from_url, create_session_factory, init_db
from app.llm.openai_compatible import OpenAICompatibleLLMClient
from app.logging_config import setup_logging

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
    await init_db(engine)

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
    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        max_context_messages=settings.max_context_messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    bot = Bot(token=settings.telegram_bot_token, session=AiohttpSession(timeout=120))
    tool_executor = ToolExecutor(bot=bot)
    dp = Dispatcher()
    dp.include_router(router)
    dp["session_factory"] = session_factory
    dp["orchestrator"] = orchestrator
    dp["tool_executor"] = tool_executor
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
