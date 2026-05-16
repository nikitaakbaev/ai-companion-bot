"""External context service placeholders."""


class NewsService:
    """Reads external context in later stages."""

    async def get_context(self) -> list[str]:
        """Return external context in later stages."""
        raise NotImplementedError

