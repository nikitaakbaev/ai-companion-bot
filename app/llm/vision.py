"""OpenAI-compatible vision client."""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx
import orjson

from app.llm.client import LLMConnectionError, LLMResponseError

logger = logging.getLogger(__name__)


class VisionClient:
    """Analyzes local images."""

    async def describe_image(self, image_path: str, prompt: str | None = None) -> str:
        """Describe an image."""
        raise NotImplementedError


class OpenAICompatibleVisionClient(VisionClient):
    """Vision client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        max_tokens: int = 500,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self._transport = transport

    async def describe_image(self, image_path: str, prompt: str | None = None) -> str:
        """Describe a local image with a multimodal model."""
        path = Path(image_path)
        if not path.exists():
            raise LLMResponseError(f"Image file does not exist: {image_path}")

        image_url = _image_data_url(path)
        user_prompt = prompt or (
            "Describe this image briefly and concretely. Mention visible people, objects, "
            "setting, mood, and any text if readable."
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    "/chat/completions",
                    content=orjson.dumps(payload),
                    headers=headers,
                )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.warning("Vision backend connection failed: %s", exc.__class__.__name__)
            raise LLMConnectionError("Vision backend is unavailable") from exc

        if response.status_code >= 400:
            logger.warning(
                "Vision request failed with HTTP %s: %s",
                response.status_code,
                response.text[:500],
            )
            raise LLMResponseError(f"Vision request failed with HTTP {response.status_code}")

        data = _decode_json(response.content)
        content = _extract_content(data)
        logger.info("Received vision response", extra={"response_length": len(content)})
        return content


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _decode_json(content: bytes) -> dict[str, Any]:
    try:
        data = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        raise LLMResponseError("Vision backend returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("Vision backend returned non-object JSON")
    return data


def _extract_content(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError("Vision response does not contain choices[0].message.content") from exc
    if not isinstance(content, str):
        raise LLMResponseError("Vision response content is not a string")
    return content.strip()
