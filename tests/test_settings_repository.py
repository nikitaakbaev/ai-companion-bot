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

