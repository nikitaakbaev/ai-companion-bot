"""Validation and compatibility logic for Stable Waifu tags."""

from app.image_generation.stable_waifu_prompting.forbidden_tags import FORBIDDEN_TAGS
from app.image_generation.stable_waifu_prompting.mood_presets import CONFLICTING_MOODS
from app.image_generation.stable_waifu_prompting.tag_cleaner import PromptCleaner

CONFLICT_GROUPS = (
    {"selfie", "mirror selfie"},
    {"portrait", "landscape"},
    {"day", "night"},
    {"warm lighting", "blue lighting", "neon lighting", "soft lighting"},
    *CONFLICTING_MOODS,
)
PORTRAIT_HINTS = ("selfie", "close-up", "upper body")
LANDSCAPE_HINTS = ("wide shot", "room composition")
SQUARE_HINTS = ("centered composition", "close framing")


class PromptValidator:
    """Validate, deduplicate, and adapt tags for Stable Waifu."""

    def __init__(
        self,
        max_tags: int = 35,
        orientation: str = "",
        aspect_ratio: str = "",
        forbidden_tags: list[str] | None = None,
    ) -> None:
        self.max_tags = max_tags
        self.orientation = orientation.strip().casefold()
        self.aspect_ratio = aspect_ratio.strip()
        self.forbidden_tags = {
            PromptCleaner().clean_tag(tag)
            for tag in (forbidden_tags if forbidden_tags is not None else FORBIDDEN_TAGS)
        }
        self.cleaner = PromptCleaner()

    def validate(self, tags: list[str]) -> list[str]:
        """Return Stable Waifu compatible tags."""
        result: list[str] = []
        seen: set[str] = set()
        for raw_tag in tags:
            tag = self.cleaner.clean_tag(raw_tag)
            if not tag or tag in self.forbidden_tags or tag in seen:
                continue
            _remove_conflicts(result, tag)
            seen = set(result)
            seen.add(tag)
            result.append(tag)
            if len(result) >= self.max_tags:
                break

        result = self._apply_orientation(result)
        result = self._apply_aspect_ratio(result)
        return result[: self.max_tags]

    def _apply_orientation(self, tags: list[str]) -> list[str]:
        if self.orientation == "portrait":
            return _append_missing(tags, PORTRAIT_HINTS, max_new=1)
        if self.orientation == "landscape":
            return _append_missing(tags, LANDSCAPE_HINTS, max_new=1)
        return tags

    def _apply_aspect_ratio(self, tags: list[str]) -> list[str]:
        if self.aspect_ratio == "9:16":
            return _append_missing(tags, ("upper body",), max_new=1)
        if self.aspect_ratio == "16:9":
            return _append_missing(tags, ("wide shot",), max_new=1)
        if self.aspect_ratio == "1:1":
            return _append_missing(tags, SQUARE_HINTS, max_new=1)
        return tags


def _append_missing(tags: list[str], candidates: tuple[str, ...], max_new: int) -> list[str]:
    result = list(tags)
    added = 0
    for candidate in candidates:
        if candidate in result:
            return result
    for candidate in candidates:
        if candidate not in result:
            result.append(candidate)
            added += 1
        if added >= max_new:
            break
    return result


def _remove_conflicts(tags: list[str], new_tag: str) -> None:
    for group in CONFLICT_GROUPS:
        if new_tag not in group:
            continue
        tags[:] = [tag for tag in tags if tag not in group]
        return
