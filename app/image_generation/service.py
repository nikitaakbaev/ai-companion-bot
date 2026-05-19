"""Provider-agnostic image generation service."""

from typing import Protocol

from app.image_generation.comfyui_client import ComfyUIClient
from app.image_generation.prompt_builder import (
    StableWaifuPromptBuilder,
    build_image_prompt,
    build_negative_prompt,
)
from app.image_generation.stable_waifu_provider import ImageGenerationResult


class ImageProvider(Protocol):
    """Provider protocol used by the image generation service."""

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> ImageGenerationResult:
        """Generate an image."""


class ComfyUIImageProvider:
    """Adapter for the local ComfyUI client."""

    def __init__(self, client: ComfyUIClient) -> None:
        self.client = client

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> ImageGenerationResult:
        """Generate an image through ComfyUI."""
        image_path = await self.client.generate_image(
            positive_prompt=prompt,
            negative_prompt=negative_prompt or "",
        )
        return ImageGenerationResult(
            provider="comfyui",
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )


class ImageGenerationService:
    """Build prompts and delegate generation to the selected provider."""

    def __init__(
        self,
        provider_name: str,
        provider: ImageProvider,
        comfy_base_positive_prompt: str,
        comfy_base_negative_prompt: str,
        stable_prompt_builder: StableWaifuPromptBuilder,
    ) -> None:
        self.provider_name = provider_name
        self.provider = provider
        self.comfy_base_positive_prompt = comfy_base_positive_prompt
        self.comfy_base_negative_prompt = comfy_base_negative_prompt
        self.stable_prompt_builder = stable_prompt_builder

    async def generate(
        self,
        scene: str,
        mood: str = "",
        style: str = "",
        negative: str = "",
    ) -> ImageGenerationResult:
        """Generate an image from scene-level context."""
        if self.provider_name == "stable_waifu_telegram":
            prompt = self.stable_prompt_builder.build_prompt(
                scene_tags=scene,
                mood=mood,
                style=style,
            )
            return await self.provider.generate(prompt.positive, prompt.negative)

        positive_prompt = build_image_prompt(
            self.comfy_base_positive_prompt,
            description=scene,
            mood=mood,
            style=style,
        )
        negative_prompt = build_negative_prompt(self.comfy_base_negative_prompt, negative)
        return await self.provider.generate(positive_prompt, negative_prompt)

    def update_base_prompt(
        self,
        add_tags: str = "",
        remove_tags: str = "",
        set_tags: str = "",
    ) -> dict[str, str]:
        """Update runtime base image prompt tags."""
        base_tags = self.stable_prompt_builder.update_base_tags(
            add_tags=add_tags,
            remove_tags=remove_tags,
            set_tags=set_tags,
        )
        if self.provider_name == "comfyui":
            self.comfy_base_positive_prompt = base_tags
        return {
            "provider": self.provider_name,
            "base_tags": base_tags,
        }
