"""Telethon userbot wrapper for talking to Telegram bots as a user."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TelegramUserbotError(Exception):
    """Raised when the Telegram userbot cannot complete an operation."""


class TelegramUserbotClient:
    """Thin async wrapper around Telethon's TelegramClient."""

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_path: str,
    ) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_path = session_path
        self._client: Any | None = None
        self._started = False

    async def start(self) -> None:
        """Start the Telethon client once for the application lifetime."""
        if self._started:
            return
        try:
            from telethon import TelegramClient
        except ModuleNotFoundError as exc:
            raise TelegramUserbotError("telethon is not installed") from exc

        session_file = Path(self.session_path)
        session_file.parent.mkdir(parents=True, exist_ok=True)
        self._client = TelegramClient(str(session_file), self.api_id, self.api_hash)
        logger.info("Starting Telegram userbot session", extra={"session_path": self.session_path})
        await self._client.start()
        self._started = True

    async def stop(self) -> None:
        """Disconnect the Telethon client."""
        if self._client is None:
            return
        await self._client.disconnect()
        self._started = False

    async def send_message(self, entity: str, text: str):
        """Send a message to an entity."""
        client = self._require_client()
        return await client.send_message(entity, text)

    async def get_recent_messages(self, entity: str, limit: int = 10):
        """Return recent messages from an entity."""
        client = self._require_client()
        return await client.get_messages(entity, limit=limit)

    async def download_media(self, message, output_dir: str):
        """Download message media to the output directory and return its path."""
        client = self._require_client()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        downloaded = await client.download_media(message, file=str(output_path))
        if downloaded:
            return downloaded

        for media in _iter_downloadable_media(message):
            downloaded = await client.download_media(media, file=str(output_path))
            if downloaded:
                return downloaded
        return None

    @property
    def is_started(self) -> bool:
        """Return whether the userbot session is started."""
        return self._started

    def _require_client(self):
        if self._client is None or not self._started:
            raise TelegramUserbotError("Telegram userbot is not started")
        return self._client


def _iter_downloadable_media(message) -> list[Any]:
    """Return nested media objects Telethon may not download from the message wrapper."""
    media_items: list[Any] = []
    for media in (
        getattr(message, "document", None),
        getattr(message, "photo", None),
        getattr(getattr(message, "media", None), "document", None),
        getattr(getattr(message, "media", None), "photo", None),
        getattr(getattr(getattr(message, "media", None), "webpage", None), "document", None),
        getattr(getattr(message, "web_preview", None), "document", None),
    ):
        if media is not None and media not in media_items:
            media_items.append(media)
    return media_items
