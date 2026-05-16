"""LLM client interface placeholders."""


class LLMClient:
    """High-level LLM client facade."""

    async def generate_text(self) -> str:
        """Generate text in later stages."""
        raise NotImplementedError

