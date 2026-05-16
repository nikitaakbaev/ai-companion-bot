"""Application entrypoint."""

import asyncio
import logging

from aiogram import Bot, Dispatcher

from app.bot.router import router
from app.config import get_settings
from app.database.session import create_engine_from_url, create_session_factory, init_db
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

    engine = create_engine_from_url(settings.database_url)
    session_factory = create_session_factory(engine)
    await init_db(engine)

    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN is not set. Telegram bot is not started.")
        await engine.dispose()
        return

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()
    dp.include_router(router)
    dp["session_factory"] = session_factory

    logger.info("Starting Telegram polling")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
