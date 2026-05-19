import httpx
import pytest

from app.llm.client import LLMResponseError
from app.llm.vision import OpenAICompatibleVisionClient


async def test_vision_client_sends_image_and_returns_description(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-jpeg")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "A blue-haired girl in a room."}}]},
        )

    client = OpenAICompatibleVisionClient(
        base_url="http://vision.test/v1",
        api_key="key",
        model="vision-model",
        transport=httpx.MockTransport(handler),
    )

    result = await client.describe_image(str(image_path), prompt="Describe")

    assert result == "A blue-haired girl in a room."
    assert b"data:image/jpeg;base64" in captured["body"]
    assert b"vision-model" in captured["body"]


async def test_vision_client_raises_on_http_error(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-jpeg")
    client = OpenAICompatibleVisionClient(
        base_url="http://vision.test/v1",
        api_key="key",
        model="vision-model",
        transport=httpx.MockTransport(lambda request: httpx.Response(400, text="bad image")),
    )

    with pytest.raises(LLMResponseError):
        await client.describe_image(str(image_path))
