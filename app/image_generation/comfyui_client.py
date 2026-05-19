"""ComfyUI API client."""

import asyncio
import logging
import random
import uuid
from pathlib import Path
from typing import Any

import httpx
import orjson

logger = logging.getLogger(__name__)


class ComfyUIError(Exception):
    """Raised when image generation fails."""


class ComfyUIClient:
    """Generates images through the ComfyUI HTTP API."""

    def __init__(
        self,
        base_url: str,
        output_dir: str,
        checkpoint: str,
        clip_skip: int = -2,
        sampler: str = "euler",
        scheduler: str = "normal",
        steps: int = 25,
        cfg: float = 7.0,
        width: int = 768,
        height: int = 1024,
        timeout_seconds: int = 120,
        poll_interval_seconds: float = 1.0,
        poll_attempts: int = 120,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.checkpoint = checkpoint
        self.clip_skip = clip_skip
        self.sampler = sampler
        self.scheduler = scheduler
        self.steps = steps
        self.cfg = cfg
        self.width = width
        self.height = height
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.poll_attempts = poll_attempts
        self._transport = transport

    async def generate_image(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int | None = None,
    ) -> str:
        """Queue a ComfyUI workflow, wait for completion, and return the local image path."""
        clean_positive = positive_prompt.strip()
        if not clean_positive:
            raise ComfyUIError("Positive prompt must not be empty")

        actual_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
        checkpoint = await self._resolve_checkpoint()
        workflow = self._build_workflow(
            clean_positive,
            negative_prompt.strip(),
            actual_seed,
            checkpoint=checkpoint,
        )
        client_id = str(uuid.uuid4())

        prompt_id = await self._queue_prompt(workflow, client_id)
        image_ref = await self._wait_for_image(prompt_id)
        return await self._download_image(image_ref, prompt_id)

    async def _queue_prompt(self, workflow: dict[str, Any], client_id: str) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        async with self._client() as client:
            response = await client.post("/prompt", content=orjson.dumps(payload))
        if response.status_code >= 400:
            raise ComfyUIError(_http_error("ComfyUI /prompt", response))

        data = _decode_json(response.content)
        prompt_id = data.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyUIError("ComfyUI response does not contain prompt_id")
        logger.info("Queued ComfyUI prompt", extra={"prompt_id": prompt_id})
        return prompt_id

    async def _wait_for_image(self, prompt_id: str) -> dict[str, str]:
        for _ in range(self.poll_attempts):
            async with self._client() as client:
                response = await client.get(f"/history/{prompt_id}")
            if response.status_code >= 400:
                raise ComfyUIError(_http_error("ComfyUI /history", response))

            data = _decode_json(response.content)
            image_ref = _find_first_image(data, prompt_id)
            if image_ref is not None:
                return image_ref
            await asyncio.sleep(self.poll_interval_seconds)

        raise ComfyUIError("ComfyUI generation timed out")

    async def _download_image(self, image_ref: dict[str, str], prompt_id: str) -> str:
        params = {
            "filename": image_ref["filename"],
            "subfolder": image_ref.get("subfolder", ""),
            "type": image_ref.get("type", "output"),
        }
        async with self._client() as client:
            response = await client.get("/view", params=params)
        if response.status_code >= 400:
            raise ComfyUIError(_http_error("ComfyUI /view", response))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        extension = Path(image_ref["filename"]).suffix or ".png"
        output_path = self.output_dir / f"{prompt_id}{extension}"
        output_path.write_bytes(response.content)
        logger.info("Downloaded generated image", extra={"path": str(output_path)})
        return str(output_path)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self._transport,
        )

    async def _resolve_checkpoint(self) -> str:
        """Resolve a checkpoint stem to the exact ComfyUI checkpoint filename when possible."""
        try:
            async with self._client() as client:
                response = await client.get("/object_info/CheckpointLoaderSimple")
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError):
            return self.checkpoint
        if response.status_code >= 400:
            return self.checkpoint

        try:
            data = _decode_json(response.content)
        except ComfyUIError:
            return self.checkpoint
        choices = _checkpoint_choices(data)
        if not choices:
            return self.checkpoint
        if self.checkpoint in choices:
            return self.checkpoint

        requested_stem = Path(self.checkpoint).stem.casefold()
        for choice in choices:
            choice_path = Path(choice)
            if choice_path.stem.casefold() == requested_stem:
                logger.info(
                    "Resolved ComfyUI checkpoint",
                    extra={"configured": self.checkpoint, "resolved": choice},
                )
                return choice
        for choice in choices:
            if choice.casefold().startswith(self.checkpoint.casefold()):
                logger.info(
                    "Resolved ComfyUI checkpoint by prefix",
                    extra={"configured": self.checkpoint, "resolved": choice},
                )
                return choice
        return self.checkpoint

    def _build_workflow(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint or self.checkpoint},
            },
            "2": {
                "class_type": "CLIPSetLastLayer",
                "inputs": {"clip": ["1", 1], "stop_at_clip_layer": self.clip_skip},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": positive_prompt, "clip": ["2", 0]},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["2", 0]},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": self.width, "height": self.height, "batch_size": 1},
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": self.steps,
                    "cfg": self.cfg,
                    "sampler_name": self.sampler,
                    "scheduler": self.scheduler,
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "ai_companion", "images": ["7", 0]},
            },
        }


def _decode_json(content: bytes) -> dict[str, Any]:
    try:
        data = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        raise ComfyUIError("ComfyUI returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ComfyUIError("ComfyUI returned non-object JSON")
    return data


def _find_first_image(data: dict[str, Any], prompt_id: str) -> dict[str, str] | None:
    prompt_data = data.get(prompt_id)
    if not isinstance(prompt_data, dict):
        return None
    outputs = prompt_data.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        images = output.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            if isinstance(filename, str) and filename:
                return {
                    "filename": filename,
                    "subfolder": str(image.get("subfolder") or ""),
                    "type": str(image.get("type") or "output"),
                }
    return None


def _checkpoint_choices(data: dict[str, Any]) -> list[str]:
    try:
        choices = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    except (KeyError, IndexError, TypeError):
        return []
    if not isinstance(choices, list):
        return []
    return [choice for choice in choices if isinstance(choice, str)]


def _http_error(label: str, response: httpx.Response) -> str:
    body = response.text.strip()
    if len(body) > 1000:
        body = body[:1000] + "..."
    return f"{label} failed with HTTP {response.status_code}: {body or '<empty body>'}"
