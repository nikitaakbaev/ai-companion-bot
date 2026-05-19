from app.memory.profiles import (
    format_profile,
    format_profiles_for_prompt,
    profiles_from_settings,
)
from app.config import Settings


def test_profiles_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        user_profile_name="Roman",
        user_profile_appearance="dark hair",
        default_character_name="Kuni",
        default_character_age="20",
    )

    user_profile, character_profile = profiles_from_settings(settings)

    assert user_profile == {"name": "Roman", "appearance": "dark hair"}
    assert character_profile == {"name": "Kuni", "age": "20"}


def test_format_profile() -> None:
    text = format_profile("Your profile", {"name": "Roman", "birthday": "May 17"})

    assert "Your profile" in text
    assert "Name: Roman" in text
    assert "Birthday: May 17" in text


def test_format_profiles_for_prompt() -> None:
    text = format_profiles_for_prompt(
        {"name": "Roman"},
        {"name": "Kuni", "age": "20"},
    )

    assert "User profile" in text
    assert "- name: Roman" in text
    assert "Companion character profile" in text
    assert "- age: 20" in text
