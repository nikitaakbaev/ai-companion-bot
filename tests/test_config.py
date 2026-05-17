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
    assert settings.agent_temperature == 0.2
    assert settings.agent_max_tokens == 2048
    assert settings.agent_context_messages == 8
    assert settings.agent_max_delay_seconds == 0.5
    assert settings.agent_typing_seconds == 0.2
    assert settings.response_verifier_enabled is False
    assert settings.response_verifier_base_url is None
    assert settings.response_verifier_api_key is None
    assert settings.response_verifier_model is None
    assert settings.response_verifier_max_tokens == 200
    assert settings.auto_create_tables is False
    assert settings.default_character_name == "Kuni"
    assert settings.message_history_limit == 50
    assert settings.max_stored_message_length == 8000
    assert settings.media_storage_dir == "./data/media"
    assert settings.diary_enabled is True
    assert settings.diary_lookback_hours == 24
    assert settings.diary_min_messages == 3
    assert settings.diary_max_messages == 100
    assert settings.diary_max_input_chars == 20000
    assert settings.diary_max_entries_per_run == 8
    assert settings.diary_reflection_model is None
    assert settings.diary_skip_if_exists_for_date is True
