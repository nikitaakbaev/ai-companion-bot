import httpx
import pytest

from app.image_generation.comfyui_client import ComfyUIClient, ComfyUIError
from app.image_generation.prompt_builder import build_image_prompt, build_negative_prompt


def test_build_image_prompt_merges_base_and_context() -> None:
    assert (
        build_image_prompt("base", description="smile", mood="happy", style="anime selfie")
        == "base, smile, happy, anime selfie"
    )
    assert build_negative_prompt("bad", "lowres") == "bad, lowres"


async def test_comfyui_client_generates_and_downloads_image(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "abc"})
        if request.url.path == "/history/abc":
            return httpx.Response(
                200,
                json={
                    "abc": {
                        "outputs": {
                            "7": {
                                "images": [
                                    {
                                        "filename": "generated.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(200, content=b"png-bytes")
        return httpx.Response(404)

    client = ComfyUIClient(
        base_url="http://comfy.test",
        output_dir=str(tmp_path),
        checkpoint="model.safetensors",
        poll_interval_seconds=0,
        transport=httpx.MockTransport(handler),
    )

    image_path = await client.generate_image("positive", "negative", seed=1)

    assert image_path.endswith("abc.png")
    assert (tmp_path / "abc.png").read_bytes() == b"png-bytes"


def test_comfyui_workflow_uses_clip_skip(tmp_path) -> None:
    client = ComfyUIClient(
        base_url="http://comfy.test",
        output_dir=str(tmp_path),
        checkpoint="model.safetensors",
        clip_skip=-2,
    )

    workflow = client._build_workflow("positive", "negative", seed=1)

    assert workflow["2"]["class_type"] == "CLIPSetLastLayer"
    assert workflow["2"]["inputs"]["stop_at_clip_layer"] == -2
    assert workflow["3"]["inputs"]["clip"] == ["2", 0]
    assert workflow["4"]["inputs"]["clip"] == ["2", 0]


async def test_comfyui_client_rejects_empty_positive_prompt(tmp_path) -> None:
    client = ComfyUIClient(
        base_url="http://comfy.test",
        output_dir=str(tmp_path),
        checkpoint="model.safetensors",
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )

    with pytest.raises(ComfyUIError):
        await client.generate_image("", "negative")


async def test_comfyui_client_resolves_checkpoint_stem(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info/CheckpointLoaderSimple":
            return httpx.Response(
                200,
                json={
                    "CheckpointLoaderSimple": {
                        "input": {
                            "required": {
                                "ckpt_name": [["waiIllustriousSDXL_v170.safetensors"]]
                            }
                        }
                    }
                },
            )
        return httpx.Response(404)

    client = ComfyUIClient(
        base_url="http://comfy.test",
        output_dir=str(tmp_path),
        checkpoint="waiIllustriousSDXL_v170",
        transport=httpx.MockTransport(handler),
    )

    assert await client._resolve_checkpoint() == "waiIllustriousSDXL_v170.safetensors"


async def test_comfyui_prompt_error_includes_body(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info/CheckpointLoaderSimple":
            return httpx.Response(404)
        if request.url.path == "/prompt":
            return httpx.Response(400, text="bad workflow")
        return httpx.Response(404)

    client = ComfyUIClient(
        base_url="http://comfy.test",
        output_dir=str(tmp_path),
        checkpoint="model.safetensors",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ComfyUIError, match="bad workflow"):
        await client.generate_image("positive", "negative")
