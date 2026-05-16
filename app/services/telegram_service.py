"""Telegram service placeholders."""


class TelegramService:
    """Wraps Telegram Bot API interactions."""

    async def send_message(self, chat_id: int, text: str) -> None:
        """Send a message in later stages."""
        raise NotImplementedError

