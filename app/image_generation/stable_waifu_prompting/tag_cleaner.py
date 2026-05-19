"""Prompt and tag normalization for Stable Waifu."""

import re

MAX_TAG_WORDS = 6
MAX_TAG_LENGTH = 64
PROSE_MARKERS = (
    " a ",
    " an ",
    " the ",
    "beautiful anime girl",
    "sitting near",
    "while",
    "with her",
    "in a ",
)


class PromptCleaner:
    """Normalize compact comma-separated anime tags."""

    def clean(self, prompt: str) -> str:
        """Return normalized comma-separated tags."""
        return ", ".join(self.tags(prompt))

    def tags(self, prompt: str | list[str] | tuple[str, ...]) -> list[str]:
        """Return normalized tag list from a prompt string or sequence."""
        raw_tags = _flatten_tags(prompt)
        tags: list[str] = []
        for raw_tag in raw_tags:
            tag = self.clean_tag(raw_tag)
            if tag:
                tags.append(tag)
        return tags

    def clean_tag(self, tag: str) -> str:
        """Normalize one tag and reject prose-like fragments."""
        cleaned = " ".join(str(tag).strip(" ,.\t\r\n\"'`").casefold().split())
        cleaned = re.sub(r"\s*,+\s*", ", ", cleaned)
        if not cleaned:
            return ""
        if any(marker in f" {cleaned} " for marker in PROSE_MARKERS):
            return ""
        if len(cleaned) > MAX_TAG_LENGTH:
            return ""
        if len(cleaned.split()) > MAX_TAG_WORDS:
            return ""
        return cleaned


def _flatten_tags(prompt: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(prompt, str):
        normalized = prompt.replace("\n", ",").replace(";", ",")
        return [part for part in normalized.split(",")]
    parts: list[str] = []
    for item in prompt:
        parts.extend(_flatten_tags(str(item)))
    return parts
