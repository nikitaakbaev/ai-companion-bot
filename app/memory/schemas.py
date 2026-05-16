"""Pydantic schemas for diary memory."""

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DiaryEntryCreate(BaseModel):
    """Data needed to create a diary entry."""

    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)
    summary: str | None = None
    facts_about_user: list[str] = Field(default_factory=list)
    facts_about_relationship: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    importance: int = Field(default=5, ge=1, le=10)
    emotion: str | None = None
    source_date: date

    @field_validator("title", "content", "summary", mode="before")
    @classmethod
    def strip_strings(cls, value: Any) -> Any:
        """Trim nullable string fields."""
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("facts_about_user", "facts_about_relationship", "topics", mode="before")
    @classmethod
    def clean_string_lists(cls, value: Any) -> list[str]:
        """Normalize string lists and drop empty items."""
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class DiaryReflectionResult(BaseModel):
    """Structured result returned by the reflection LLM."""

    entries: list[DiaryEntryCreate] = Field(default_factory=list)
    day_summary: str | None = None

    @field_validator("day_summary", mode="before")
    @classmethod
    def strip_day_summary(cls, value: Any) -> Any:
        """Trim day summary."""
        if isinstance(value, str):
            return value.strip() or None
        return value
