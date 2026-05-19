"""Application configuration."""

from typing import Any
from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI Companion Bot"
    app_env: str = "local"
    debug: bool = True
    log_level: str = "INFO"
    auto_create_tables: bool = False

    telegram_bot_token: str | None = None

    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "qwen/qwen3"
    llm_timeout_seconds: int = 120
    llm_temperature: float = 0.7
    llm_max_tokens: int = 800
    llm_disable_thinking: bool = False
    agent_temperature: float = 0.2
    agent_max_tokens: int = 2048
    agent_context_messages: int = 8
    agent_user_prompt_mode: bool = False
    agent_response_format: bool = False
    agent_max_delay_seconds: float = 0.5
    agent_typing_seconds: float = 0.2
    vision_enabled: bool = True
    vision_base_url: str | None = None
    vision_api_key: str | None = None
    vision_model: str = "qwen-vl"
    vision_timeout_seconds: int = 120
    vision_max_tokens: int = 500
    vision_prompt: str = (
        "Describe this image briefly and concretely. Mention visible people, objects, "
        "setting, mood, and any text if readable."
    )
    embedding_provider: str = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    database_url: str = "sqlite+aiosqlite:///./data/bot.db"

    default_character_name: str = "Kuni"
    default_character_age: str | None = None
    default_character_birthday: str | None = None
    default_character_appearance: str | None = None
    default_character_about: str | None = None
    default_character_description: str = "AI companion with natural Telegram behavior"
    default_personality_style: str = "warm, playful, concise"

    user_profile_name: str | None = None
    user_profile_age: str | None = None
    user_profile_birthday: str | None = None
    user_profile_appearance: str | None = None
    user_profile_about: str | None = None

    default_proactive_enabled: bool = False
    default_proactive_min_interval_minutes: int = 60
    default_proactive_max_interval_minutes: int = 180

    message_history_limit: int = 50
    max_stored_message_length: int = 8000

    rag_top_k: int = 5
    rag_min_score: float = 0.65
    rag_enabled: bool = True
    rag_max_context_chars: int = 4000
    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/chroma"
    vector_collection_name: str = "diary_entries"
    max_context_messages: int = 20

    comfyui_base_url: str = "http://127.0.0.1:8188"
    comfyui_timeout_seconds: int = 120
    comfyui_poll_interval_seconds: float = 1.0
    comfyui_poll_attempts: int = 120
    comfyui_checkpoint: str = "model.safetensors"
    comfyui_clip_skip: int = -2
    comfyui_sampler: str = "euler"
    comfyui_scheduler: str = "normal"
    comfyui_steps: int = 25
    comfyui_cfg: float = 7.0
    comfyui_width: int = 768
    comfyui_height: int = 1024
    image_generation_provider: str = "comfyui"
    image_generation_enabled: bool = True
    image_base_positive_prompt: str = (
        "masterpiece, best quality, 1girl, long blue hair, blue eyes, pale skin, "
        "black hoodie, (boobs:0.4), dark cyber room, night"
    )
    image_base_negative_prompt: str = "bad quality,worst quality,worst detail,sketch,censor"
    stable_waifu_enabled: bool = False
    stable_waifu_bot_username: str = "StableWaifuBot"
    stable_waifu_model: str = "SkyVXL"
    stable_waifu_preset: str = "Opus"
    stable_waifu_timeout_seconds: int = 240
    stable_waifu_poll_interval_seconds: int = 3
    stable_waifu_click_start_button: bool = True
    stable_waifu_click_save_button: bool = True
    stable_waifu_output_dir: str = "./data/generated_images/stable_waifu"
    stable_waifu_orientation: str = "landscape"
    stable_waifu_aspect_ratio: str = "16:9"
    stable_waifu_model_search_max_page_switches: int = 10
    stable_waifu_landscape_keywords: str = "Ландшафт,Landscape"
    stable_waifu_portrait_keywords: str = "Портрет,Portrait"
    stable_waifu_model_menu_keywords: str = "Модель,Model"
    stable_waifu_preset_menu_keywords: str = "Пресет,Preset"
    stable_waifu_aspect_ratio_keywords: str = "Соотношение,Aspect,Ratio"
    stable_waifu_next_page_keywords: str = ">>,››,➡,Next"
    stable_waifu_previous_page_keywords: str = "<<,‹‹,⬅,Back"
    stable_waifu_start_keywords: str = "Начать,Start,🎉"
    stable_waifu_save_keywords: str = "Сохранить,Save"
    stable_waifu_editor_wait_keywords: str = "Новый арт,Проверь параметры,Модель,Пресет"
    stable_waifu_default_landscape_ratio: str = "16:9"
    stable_waifu_default_portrait_ratio: str = "9:16"
    telegram_userbot_api_id: int | None = None
    telegram_userbot_api_hash: str | None = None
    telegram_userbot_session_path: str = "./data/userbot/stable_waifu.session"
    image_base_tags: str = (
        "1girl, pale skin, black hoodie, oversized hoodie, blue eyes, long dark blue hair, "
        "blue gradient hair, bangs, blue nails, soft smile, cozy dark room, night, "
        "cyber aesthetic, anime, masterpiece, best quality"
    )
    image_negative_tags: str = (
        "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, "
        "missing fingers, extra limbs, deformed body, mutated hands, poorly drawn face, "
        "ugly, text, watermark, logo, cropped, jpeg artifacts, duplicate, distorted eyes, "
        "cross-eyed, bad proportions, messy background, low detail"
    )
    image_prompt_max_length: int = 900
    stable_waifu_use_pony_prefixes: bool = False
    stable_waifu_nsfw_level: int = 0
    stable_waifu_max_prompt_tags: int = 35
    generated_images_dir: str = "./data/generated_images"
    media_storage_dir: str = "./data/media"
    diary_enabled: bool = True
    diary_lookback_hours: int = 24
    diary_min_messages: int = 3
    diary_max_messages: int = 100
    diary_max_input_chars: int = 20000
    diary_max_entries_per_run: int = 8
    diary_max_tokens: int = 800
    diary_reflection_model: str | None = None
    diary_user_prompt_mode: bool = True
    diary_skip_if_exists_for_date: bool = True
    diary_auto_create_enabled: bool = True

    proactive_enabled: bool = False
    proactive_min_interval_minutes: int = 60
    proactive_max_interval_minutes: int = 180
    default_timezone: str = "Europe/Moscow"

    silent_hours_start: str = "23:00"
    silent_hours_end: str = "09:00"

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> bool:
        """Parse DEBUG defensively because host environments often reuse this variable."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "production", "prod"}:
                return False
        return bool(value)

    @model_validator(mode="after")
    def validate_image_provider_settings(self) -> "Settings":
        """Validate provider-specific image generation settings."""
        provider = self.image_generation_provider.strip().lower()
        if provider == "stable_waifu_telegram" and self.stable_waifu_enabled:
            if self.telegram_userbot_api_id is None:
                raise ValueError(
                    "TELEGRAM_USERBOT_API_ID is required for stable_waifu_telegram"
                )
            if not self.telegram_userbot_api_hash:
                raise ValueError(
                    "TELEGRAM_USERBOT_API_HASH is required for stable_waifu_telegram"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
