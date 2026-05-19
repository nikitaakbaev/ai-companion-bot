"""Image generation concurrency controls."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


class ImageGenerationQueue:
    """Serializes image generations for providers that cannot run in parallel."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run_exclusive(self, coro: Awaitable[T]) -> T:
        """Run one generation coroutine at a time."""
        async with self._lock:
            return await coro
