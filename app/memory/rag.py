"""RAG memory placeholders."""


class RAGService:
    """Retrieves relevant diary memories."""

    async def search(self, query: str) -> list[dict]:
        """Search memory in later stages."""
        raise NotImplementedError

