import json

import httpx
import pytest

from app.llm.client import ChatMessage, LLMConnectionError, LLMResponseError
from app.llm.openai_compatible import OpenAICompatibleLLMClient


async def test_client_posts_to_chat_completions_and_parses_response() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        payload = request.content.decode("utf-8")
        assert '"model":"test-model"' in payload
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
        )

    client = OpenAICompatibleLLMClient(
        base_url="http://llm.test/v1/",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    response = await client.generate_text([ChatMessage(role="user", content="Hi")])

    assert response.content == "hello"
    assert len(requests) == 1


async def test_client_raises_response_error_without_choices() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    client = OpenAICompatibleLLMClient(
        base_url="http://llm.test/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMResponseError):
        await client.generate_text([ChatMessage(role="user", content="Hi")])


async def test_client_allows_empty_content() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})

    client = OpenAICompatibleLLMClient(
        base_url="http://llm.test/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    response = await client.generate_text([ChatMessage(role="user", content="Hi")])

    assert response.content == ""


async def test_client_raises_connection_error_on_connect_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("cannot connect", request=request)

    client = OpenAICompatibleLLMClient(
        base_url="http://llm.test/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMConnectionError):
        await client.generate_text([ChatMessage(role="user", content="Hi")])


async def test_client_does_not_add_double_slash_to_url() -> None:
    seen_url = ""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_url
        seen_url = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatibleLLMClient(
        base_url="http://llm.test/v1/",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    await client.generate_text([ChatMessage(role="user", content="Hi")])

    assert seen_url == "http://llm.test/v1/chat/completions"


async def test_client_adds_response_format_for_json_mode() -> None:
    seen_payload: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_payload
        seen_payload = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = OpenAICompatibleLLMClient(
        base_url="http://llm.test/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    await client.generate_text([ChatMessage(role="user", content="Return JSON")], json_mode=True)

    assert seen_payload["response_format"] == {"type": "json_object"}


async def test_client_retries_without_json_mode_when_backend_rejects_response_format() -> None:
    payloads: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(400, text="unsupported response_format")
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = OpenAICompatibleLLMClient(
        base_url="http://llm.test/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    response = await client.generate_text(
        [ChatMessage(role="user", content="Return JSON")],
        json_mode=True,
    )

    assert response.content == "{}"
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in payloads[1]
