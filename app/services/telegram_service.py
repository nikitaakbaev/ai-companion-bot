"""Telegram service wrapper."""

from aiogram import Bot
from aiogram.types import Message


class TelegramService:
    """Wraps Telegram Bot API interactions."""

    def __init__(self, bot: Bot) -> None:
        self.bot = bot

    async def send_text(self, chat_id: int, text: str) -> Message:
        """Send a text message."""
        return await self.bot.send_message(chat_id=chat_id, text=text)
