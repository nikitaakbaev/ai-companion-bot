"""Structured Stable Waifu prompt builder."""

from dataclasses import dataclass
from typing import Any

from app.image_generation.stable_waifu_prompting.forbidden_tags import FORBIDDEN_TAGS
from app.image_generation.stable_waifu_prompting.nsfw_levels import NSFW_LEVELS
from app.image_generation.stable_waifu_prompting.scene_archetypes import SCENE_ARCHETYPES
from app.image_generation.stable_waifu_prompting.tag_cleaner import PromptCleaner
from app.image_generation.stable_waifu_prompting.validation import PromptValidator

QUALITY_PREFIXES = ["masterpiece", "best quality", "anime"]
PONY_PREFIXES = ["score_9", "score_8_up", "score_7_up"]
TAG_CATEGORY_KEYS = (
    "scene_tags",
    "emotion_tags",
    "environment_tags",
    "lighting_tags",
    "camera_tags",
    "pose_tags",
    "outfit_modifiers",
    "mood",
    "style",
)


@dataclass(frozen=True)
class StableWaifuPrompt:
    """Positive and negative tag prompts for Stable Waifu."""

    positive: str
    negative: str


class StableWaifuPromptBuilder:
    """Build final Stable Waifu prompts from structured tag categories."""

    def __init__(
        self,
        base_tags: str,
        negative_tags: str,
        max_length: int,
        orientation: str = "",
        aspect_ratio: str = "",
        use_pony_prefixes: bool = False,
        nsfw_level: int = 0,
        max_tags: int = 35,
    ) -> None:
        self.base_tags = base_tags
        self.negative_tags = negative_tags
        self.max_length = max_length
        self.orientation = orientation
        self.aspect_ratio = aspect_ratio
        self.use_pony_prefixes = use_pony_prefixes
        self.nsfw_level = nsfw_level
        self.max_tags = max_tags
        self.cleaner = PromptCleaner()

    def build_prompt(
        self,
        generated_tags: dict[str, Any] | None = None,
        scene_tags: str = "",
        mood: str | None = None,
        style: str | None = None,
    ) -> StableWaifuPrompt:
        """Return final positive/negative Stable Waifu prompts."""
        structured_tags = generated_tags or {
            "scene_tags": scene_tags,
            "emotion_tags": mood or "",
            "camera_tags": style or "",
        }
        positive_tags = self._build_positive_tags(structured_tags)
        negative_tags = self._validator().validate(self.cleaner.tags(self.negative_tags))
        return StableWaifuPrompt(
            positive=_limit_prompt(", ".join(positive_tags), self.max_length),
            negative=_limit_prompt(", ".join(negative_tags), self.max_length),
        )

    def update_base_tags(
        self,
        add_tags: str = "",
        remove_tags: str = "",
        set_tags: str = "",
    ) -> str:
        """Update runtime base tags and return the normalized base prompt."""
        validator = self._validator()
        if set_tags.strip():
            self.base_tags = ", ".join(validator.validate(self.cleaner.tags(set_tags)))
            return self.base_tags

        current_tags = self.cleaner.tags(self.base_tags)
        remove_keys = set(self.cleaner.tags(remove_tags))
        kept_tags = [tag for tag in current_tags if tag not in remove_keys]
        self.base_tags = ", ".join(validator.validate([*kept_tags, *self.cleaner.tags(add_tags)]))
        return self.base_tags

    def _build_positive_tags(self, generated_tags: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        tags.extend(PONY_PREFIXES if self.use_pony_prefixes else QUALITY_PREFIXES)
        tags.extend(self.cleaner.tags(self.base_tags))
        tags.extend(self._archetype_tags(generated_tags))
        for key in TAG_CATEGORY_KEYS:
            tags.extend(self.cleaner.tags(_value_to_tags(generated_tags.get(key))))
        tags.extend(self._nsfw_tags())
        return self._validator().validate(tags)

    def _archetype_tags(self, generated_tags: dict[str, Any]) -> list[str]:
        archetype = str(generated_tags.get("archetype") or "").strip().casefold()
        if not archetype:
            return []
        return SCENE_ARCHETYPES.get(archetype, [])

    def _validator(self) -> PromptValidator:
        return PromptValidator(
            max_tags=self.max_tags,
            orientation=self.orientation,
            aspect_ratio=self.aspect_ratio,
            forbidden_tags=FORBIDDEN_TAGS,
        )

    def _nsfw_tags(self) -> list[str]:
        tags: list[str] = []
        for level in sorted(NSFW_LEVELS):
            if level <= 0 or level > self.nsfw_level:
                continue
            tags.extend(NSFW_LEVELS[level])
        return tags


def _value_to_tags(value: Any) -> str | list[str]:
    if value is None:
        return ""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return str(value)


def _limit_prompt(prompt: str, max_length: int) -> str:
    if max_length <= 0 or len(prompt) <= max_length:
        return prompt

    tags = PromptCleaner().tags(prompt)
    result: list[str] = []
    length = 0
    for tag in tags:
        addition = len(tag) if not result else len(tag) + 2
        if length + addition > max_length:
            break
        result.append(tag)
        length += addition
    return ", ".join(result)
