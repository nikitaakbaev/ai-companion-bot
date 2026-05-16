"""Typing indicator helpers."""

import asyncio

from aiogram import Bot


async def send_typing(bot: Bot, chat_id: int, seconds: float = 1.0) -> None:
    """Send a Telegram typing action and wait for a short delay."""
    await bot.send_chat_action(chat_id=chat_id, action="typing")
    if seconds > 0:
        await asyncio.sleep(seconds)
