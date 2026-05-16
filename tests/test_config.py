from app.config import Settings


def test_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("DEBUG", raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_name == "AI Companion Bot"
    assert settings.app_env == "local"
    assert settings.llm_base_url == "http://127.0.0.1:1234/v1"
    assert settings.rag_top_k == 5
    assert settings.rag_min_score == 0.65
    assert settings.proactive_enabled is False
    assert settings.telegram_bot_token is None
    assert settings.llm_timeout_seconds == 120
    assert settings.llm_temperature == 0.7
    assert settings.llm_max_tokens == 800
    assert settings.auto_create_tables is False
    assert settings.default_character_name == "Kuni"
    assert settings.message_history_limit == 50
    assert settings.max_stored_message_length == 8000
    assert settings.media_storage_dir == "./data/media"
    assert settings.diary_enabled is False
