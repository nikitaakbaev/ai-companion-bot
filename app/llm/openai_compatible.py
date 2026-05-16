"""OpenAI-compatible API client placeholders."""


class OpenAICompatibleClient:
    """Client for OpenAI-compatible chat completion APIs."""

    async def generate_text(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Generate text using an OpenAI-compatible backend in later stages."""
        raise NotImplementedError

