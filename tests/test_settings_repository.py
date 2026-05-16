from app.config import Settings
from app.database.repositories import (
    get_bot_settings,
    get_or_create_bot_settings,
    get_or_create_user,
    update_bot_settings,
)


async def test_get_or_create_bot_settings_creates_defaults(session_factory) -> None:
    settings = Settings(_env_file=None)
    async with session_factory() as session:
        user = await get_or_create_user(session, 123, None, None, None)
        bot_settings = await get_or_create_bot_settings(session, user.id, settings)

        assert bot_settings.character_name == settings.default_character_name
        assert bot_settings.llm_model == settings.llm_model


async def test_get_or_create_bot_settings_does_not_duplicate(session_factory) -> None:
    settings = Settings(_env_file=None)
    async with session_factory() as session:
        user = await get_or_create_user(session, 123, None, None, None)
        first = await get_or_create_bot_settings(session, user.id, settings)
        second = await get_or_create_bot_settings(session, user.id, settings)

        assert first.id == second.id


async def test_update_bot_settings(session_factory) -> None:
    settings = Settings(_env_file=None)
    async with session_factory() as session:
        user = await get_or_create_user(session, 123, None, None, None)
        await get_or_create_bot_settings(session, user.id, settings)
        updated = await update_bot_settings(session, user.id, character_name="Elena")
        fetched = await get_bot_settings(session, user.id)

        assert updated.character_name == "Elena"
        assert fetched is not None
        assert fetched.character_name == "Elena"


async def test_get_or_create_bot_settings_syncs_env_defaults(session_factory) -> None:
    initial = Settings(_env_file=None)
    updated = Settings(
        _env_file=None,
        default_character_name="Mira",
        default_character_description="Strictly follow this character description.",
        default_personality_style="brief and direct",
        llm_model="new-model",
    )
    async with session_factory() as session:
        user = await get_or_create_user(session, 123, None, None, None)
        await get_or_create_bot_settings(session, user.id, initial)
        synced = await get_or_create_bot_settings(session, user.id, updated)

        assert synced.character_name == "Mira"
        assert synced.character_description == "Strictly follow this character description."
        assert synced.personality_style == "brief and direct"
        assert synced.llm_model == "new-model"
