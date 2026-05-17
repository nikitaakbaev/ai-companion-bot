"""LLM client interfaces and common schemas."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    """One chat message sent to an LLM backend."""

    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Text response returned by an LLM backend."""

    content: str
    raw: dict | None = None


class LLMClientError(Exception):
    """Base error for LLM client failures."""


class LLMConnectionError(LLMClientError):
    """Raised when the LLM backend cannot be reached or returns a retriable error."""


class LLMResponseError(LLMClientError):
    """Raised when the LLM backend returns an invalid or non-retriable response."""


class LLMClient:
    """High-level LLM client interface."""

    async def generate_text(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate text from chat messages."""
        raise NotImplementedError
