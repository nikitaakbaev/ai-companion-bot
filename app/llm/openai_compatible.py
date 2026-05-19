"""OpenAI-compatible LLM client."""

import asyncio
import json
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
        disable_thinking: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.disable_thinking = disable_thinking
        self._transport = transport
        self._request_lock = asyncio.Lock()

    async def generate_text(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate text using an OpenAI-compatible backend."""
        payload = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if json_mode and response_format is None:
            response_format = {"type": "json_object"}
        if response_format is not None:
            payload["response_format"] = response_format

        logger.info(
            "Sending request to LLM",
            extra={"model": self.model, "message_count": len(payload["messages"])},
        )
        async with self._request_lock:
            try:
                data = await self._post_chat_completions(payload)
            except LLMResponseError as exc:
                if json_mode and "response_format" in payload and _is_response_format_error(exc):
                    logger.warning("LLM backend rejected response_format; retrying without JSON mode")
                    payload.pop("response_format", None)
                    data = await self._post_chat_completions(payload)
                else:
                    raise
        content = self._extract_content(data)
        finish_reason = _extract_finish_reason(data)
        logger.info("Received response from LLM", extra={"response_length": len(content)})
        return LLMResponse(content=content, raw=data, finish_reason=finish_reason)

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        payload_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.content
        ]
        if not self.disable_thinking:
            return payload_messages

        for message in reversed(payload_messages):
            if message["role"] != "user":
                continue
            content = message["content"].rstrip()
            if "/no_think" not in content:
                message["content"] = f"{content}\n/no_think"
            break
        return payload_messages

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
            body = response.text[:500]
            logger.error(
                "LLM request failed with HTTP %s: %s",
                response.status_code,
                body,
            )
            raise LLMResponseError(f"LLM request failed with HTTP {response.status_code}: {body}")

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
        if content.strip():
            return content.strip()

        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            structured_content = _extract_structured_content(reasoning_content)
            if structured_content:
                logger.warning(
                    "LLM response content is empty; extracted structured JSON from reasoning_content"
                )
                return structured_content
            logger.warning(
                "LLM response content is empty; ignoring reasoning_content because it is not user output"
            )
        else:
            logger.warning("LLM response content is empty")
        return ""


def _extract_structured_content(text: str) -> str:
    """Extract a complete structured JSON object from model reasoning, if present."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        if _looks_like_supported_structured_response(candidate):
            return json.dumps(candidate, ensure_ascii=False)
    return ""


def _looks_like_supported_structured_response(candidate: dict[str, Any]) -> bool:
    agent_keys = {"thought", "action", "messages", "tool_input", "emotion", "delay_seconds"}
    diary_keys = {"entries", "day_summary"}
    return agent_keys.issubset(candidate.keys()) or bool(diary_keys & candidate.keys())


def _extract_finish_reason(data: dict) -> str | None:
    try:
        finish_reason = data["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return finish_reason if isinstance(finish_reason, str) else None


def _is_response_format_error(exc: LLMResponseError) -> bool:
    return "response_format" in str(exc).casefold()
