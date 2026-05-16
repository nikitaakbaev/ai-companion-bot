"""Vector store placeholders."""


class VectorStore:
    """Stores and searches embeddings."""

    async def add(self, text: str, metadata: dict) -> str:
        """Add an item in later stages."""
        raise NotImplementedError

