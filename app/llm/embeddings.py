"""Embedding client placeholders."""


class EmbeddingClient:
    """Creates text embeddings."""

    async def embed_text(self, text: str) -> list[float]:
        """Create an embedding in later stages."""
        raise NotImplementedError

