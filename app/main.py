"""Stage 1 application entrypoint."""

import logging

from app.config import get_settings
from app.logging_config import setup_logging


def main() -> None:
    """Start the stage 1 bootstrap."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting %s", settings.app_name)
    logger.info("Environment: %s", settings.app_env)
    logger.info("Debug: %s", settings.debug)
    logger.info("Database URL: %s", settings.database_url)
    logger.info("LLM base URL: %s", settings.llm_base_url)
    logger.info("LLM model: %s", settings.llm_model)

    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN is not set. Telegram bot is not started in stage 1.")

    logger.info("Stage 1 bootstrap completed successfully.")


if __name__ == "__main__":
    main()

