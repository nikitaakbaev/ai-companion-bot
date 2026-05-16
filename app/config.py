"""Application configuration."""

from typing import Any
from functools import lru_cache

from pydantic import field_validator
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

    telegram_bot_token: str | None = None

    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "qwen/qwen3"
    llm_timeout_seconds: int = 120
    llm_temperature: float = 0.7
    llm_max_tokens: int = 800
    vision_model: str = "qwen-vl"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    database_url: str = "sqlite+aiosqlite:///./data/bot.db"

    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/chroma"
    rag_top_k: int = 5
    rag_min_score: float = 0.65
    max_context_messages: int = 20

    comfyui_base_url: str = "http://127.0.0.1:8188"
    generated_images_dir: str = "./data/generated_images"

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


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
