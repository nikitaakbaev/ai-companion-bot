"""Database session placeholders."""


async def get_session() -> None:
    """Yield an async database session in later stages."""
    raise NotImplementedError

