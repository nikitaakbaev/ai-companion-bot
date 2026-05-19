"""Prompt helpers for generated companion images.

Stable Waifu prompting lives in app.image_generation.stable_waifu_prompting.
This module keeps the old imports stable for the rest of the app.
"""

from app.image_generation.stable_waifu_prompting.prompt_builder import (
    StableWaifuPrompt,
    StableWaifuPromptBuilder,
)


def build_image_prompt(
    base_prompt: str,
    description: str = "",
    mood: str = "",
    style: str = "",
) -> str:
    """Merge the fixed character prompt with contextual LLM additions."""
    return ", ".join(_clean_parts(base_prompt, description, mood, style))


def build_negative_prompt(base_prompt: str, extra_negative: str = "") -> str:
    """Merge the fixed negative prompt with optional contextual negatives."""
    return ", ".join(_clean_parts(base_prompt, extra_negative))


def _clean_parts(*parts: str) -> list[str]:
    return [part.strip(" ,") for part in parts if part and part.strip(" ,")]


__all__ = [
    "StableWaifuPrompt",
    "StableWaifuPromptBuilder",
    "build_image_prompt",
    "build_negative_prompt",
]
