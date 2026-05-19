"""Structured profile helpers for the user and companion character."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings

PROFILE_FIELDS = ("name", "age", "birthday", "appearance", "about")
def profiles_from_settings(settings: "Settings") -> tuple[dict[str, str], dict[str, str]]:
    """Build user and character profiles from environment-backed settings."""
    user_profile = _clean_profile(
        {
            "name": settings.user_profile_name,
            "age": settings.user_profile_age,
            "birthday": settings.user_profile_birthday,
            "appearance": settings.user_profile_appearance,
            "about": settings.user_profile_about,
        }
    )
    character_profile = _clean_profile(
        {
            "name": settings.default_character_name,
            "age": settings.default_character_age,
            "birthday": settings.default_character_birthday,
            "appearance": settings.default_character_appearance,
            "about": settings.default_character_about,
        }
    )
    return user_profile, character_profile


def format_profile(title: str, profile: dict[str, str]) -> str:
    """Format a structured profile for Telegram."""
    if not profile:
        return f"{title}:\n\nProfile is empty."

    labels = {
        "name": "Name",
        "age": "Age",
        "birthday": "Birthday",
        "appearance": "Appearance",
        "about": "About",
    }
    lines = [f"{title}:\n"]
    for field in PROFILE_FIELDS:
        value = profile.get(field)
        if value:
            lines.append(f"{labels[field]}: {value}")
    return "\n".join(lines)


def format_profiles_for_prompt(
    user_profile: dict[str, str],
    character_profile: dict[str, str],
) -> str:
    """Format profiles for the agent prompt."""
    blocks = []
    if user_profile:
        blocks.append(_prompt_block("User profile", user_profile))
    if character_profile:
        blocks.append(_prompt_block("Companion character profile", character_profile))
    return "\n\n".join(blocks) if blocks else "No structured profiles configured."


def _prompt_block(title: str, profile: dict[str, str]) -> str:
    lines = [f"{title}:"]
    for field in PROFILE_FIELDS:
        value = profile.get(field)
        if value:
            lines.append(f"- {field}: {value}")
    return "\n".join(lines)


def _clean_profile(profile: dict[str, str | None]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for field in PROFILE_FIELDS:
        value = profile.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            clean[field] = text
    return clean
