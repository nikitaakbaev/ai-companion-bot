from app.config import Settings


def test_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("DEBUG", raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_name == "AI Companion Bot"
    assert settings.app_env == "local"
    assert settings.llm_base_url == "http://127.0.0.1:1234/v1"
    assert settings.rag_top_k == 5
    assert settings.rag_min_score == 0.65
    assert settings.rag_enabled is True
    assert settings.rag_max_context_chars == 4000
    assert settings.embedding_provider == "auto"
    assert settings.embedding_dimensions == 384
    assert settings.vector_store_type == "chroma"
    assert settings.vector_store_path == "./data/chroma"
    assert settings.vector_collection_name == "diary_entries"
    assert settings.proactive_enabled is False
    assert settings.telegram_bot_token is None
    assert settings.llm_timeout_seconds == 120
    assert settings.llm_temperature == 0.7
    assert settings.llm_max_tokens == 800
    assert settings.llm_disable_thinking is False
    assert settings.agent_temperature == 0.2
    assert settings.agent_max_tokens == 2048
    assert settings.agent_context_messages == 8
    assert settings.agent_user_prompt_mode is False
    assert settings.agent_response_format is False
    assert settings.agent_max_delay_seconds == 0.5
    assert settings.agent_typing_seconds == 0.2
    assert settings.vision_enabled is True
    assert settings.vision_base_url is None
    assert settings.vision_api_key is None
    assert settings.vision_timeout_seconds == 120
    assert settings.vision_max_tokens == 500
    assert settings.auto_create_tables is False
    assert settings.default_character_name == "Kuni"
    assert settings.default_character_age is None
    assert settings.default_character_birthday is None
    assert settings.default_character_appearance is None
    assert settings.default_character_about is None
    assert settings.user_profile_name is None
    assert settings.message_history_limit == 50
    assert settings.max_stored_message_length == 8000
    assert settings.media_storage_dir == "./data/media"
    assert settings.image_generation_enabled is True
    assert settings.image_generation_provider == "comfyui"
    assert settings.comfyui_base_url == "http://127.0.0.1:8188"
    assert settings.comfyui_checkpoint == "model.safetensors"
    assert settings.comfyui_clip_skip == -2
    assert settings.comfyui_width == 768
    assert settings.comfyui_height == 1024
    assert "long blue hair" in settings.image_base_positive_prompt
    assert "bad quality" in settings.image_base_negative_prompt
    assert settings.stable_waifu_enabled is False
    assert settings.stable_waifu_bot_username == "StableWaifuBot"
    assert settings.stable_waifu_model == "SkyVXL"
    assert settings.stable_waifu_preset == "Opus"
    assert settings.stable_waifu_timeout_seconds == 240
    assert settings.stable_waifu_poll_interval_seconds == 3
    assert settings.stable_waifu_orientation == "landscape"
    assert settings.stable_waifu_aspect_ratio == "16:9"
    assert settings.stable_waifu_model_search_max_page_switches == 10
    assert settings.stable_waifu_landscape_keywords == "Ландшафт,Landscape"
    assert settings.stable_waifu_portrait_keywords == "Портрет,Portrait"
    assert settings.stable_waifu_model_menu_keywords == "Модель,Model"
    assert settings.stable_waifu_preset_menu_keywords == "Пресет,Preset"
    assert settings.stable_waifu_aspect_ratio_keywords == "Соотношение,Aspect,Ratio"
    assert settings.stable_waifu_next_page_keywords == ">>,››,➡,Next"
    assert settings.stable_waifu_previous_page_keywords == "<<,‹‹,⬅,Back"
    assert settings.stable_waifu_start_keywords == "Начать,Start,🎉"
    assert settings.stable_waifu_save_keywords == "Сохранить,Save"
    assert settings.stable_waifu_editor_wait_keywords == "Новый арт,Проверь параметры,Модель,Пресет"
    assert settings.stable_waifu_default_landscape_ratio == "16:9"
    assert settings.stable_waifu_default_portrait_ratio == "9:16"
    assert settings.telegram_userbot_api_id is None
    assert settings.telegram_userbot_api_hash is None
    assert settings.telegram_userbot_session_path == "./data/userbot/stable_waifu.session"
    assert "long dark blue hair" in settings.image_base_tags
    assert "bad anatomy" in settings.image_negative_tags
    assert settings.image_prompt_max_length == 900
    assert settings.stable_waifu_use_pony_prefixes is False
    assert settings.stable_waifu_nsfw_level == 0
    assert settings.stable_waifu_max_prompt_tags == 35
    assert settings.diary_enabled is True
    assert settings.diary_lookback_hours == 24
    assert settings.diary_min_messages == 3
    assert settings.diary_max_messages == 100
    assert settings.diary_max_input_chars == 20000
    assert settings.diary_max_entries_per_run == 8
    assert settings.diary_max_tokens == 800
    assert settings.diary_reflection_model is None
    assert settings.diary_user_prompt_mode is True
    assert settings.diary_skip_if_exists_for_date is True
    assert settings.diary_auto_create_enabled is True


def test_stable_waifu_requires_userbot_credentials(monkeypatch) -> None:
    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "stable_waifu_telegram")
    monkeypatch.setenv("STABLE_WAIFU_ENABLED", "true")
    monkeypatch.delenv("TELEGRAM_USERBOT_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_USERBOT_API_HASH", raising=False)

    try:
        Settings(_env_file=None)
    except ValueError as exc:
        assert "TELEGRAM_USERBOT_API_ID" in str(exc)
    else:
        raise AssertionError("Stable Waifu provider should require Telethon credentials")
