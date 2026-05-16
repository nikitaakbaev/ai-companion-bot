"""OpenAI-compatible LLM client."""

import logging
from typing import Any

import httpx
import orjson
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.llm.client import (
    ChatMessage,
    LLMClient,
    LLMConnectionError,
    LLMResponse,
    LLMResponseError,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleLLMClient(LLMClient):
    """Client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        default_temperature: float = 0.7,
        default_max_tokens: int = 800,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._transport = transport

    async def generate_text(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text using an OpenAI-compatible backend."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
                if message.content
            ],
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }

        logger.info(
            "Sending request to LLM",
            extra={"model": self.model, "message_count": len(payload["messages"])},
        )
        data = await self._post_chat_completions(payload)
        content = self._extract_content(data)
        logger.info("Received response from LLM", extra={"response_length": len(content)})
        return LLMResponse(content=content, raw=data)

    @retry(
        retry=retry_if_exception_type(LLMConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _post_chat_completions(self, payload: dict[str, Any]) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    url,
                    content=orjson.dumps(payload),
                    headers=headers,
                )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.warning("LLM connection failed: %s", exc.__class__.__name__)
            raise LLMConnectionError("LLM backend is unavailable") from exc

        if response.status_code >= 500:
            logger.warning("LLM backend returned HTTP %s", response.status_code)
            raise LLMConnectionError(f"LLM backend returned HTTP {response.status_code}")
        if response.status_code >= 400:
            logger.error("LLM request failed with HTTP %s", response.status_code)
            raise LLMResponseError(f"LLM request failed with HTTP {response.status_code}")

        try:
            data = orjson.loads(response.content)
        except orjson.JSONDecodeError as exc:
            raise LLMResponseError("LLM backend returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise LLMResponseError("LLM backend returned a non-object response")
        return data

    @staticmethod
    def _extract_content(data: dict) -> str:
        try:
            choices = data["choices"]
            message = choices[0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM response does not contain choices[0].message.content") from exc

        if not isinstance(content, str):
            raise LLMResponseError("LLM response content is not a string")
        if not content.strip():
            raise LLMResponseError("LLM response content is empty")
        return content.strip()
