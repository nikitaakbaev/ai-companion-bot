"""Diary reflection JSON parser."""

from pydantic import ValidationError

from app.memory.schemas import DiaryReflectionResult
from app.utils.json import JsonObjectParseError, loads_json_object


class DiaryReflectionParseError(Exception):
    """Raised when diary reflection JSON cannot be parsed."""


def parse_diary_reflection(raw_text: str) -> DiaryReflectionResult:
    """Parse diary reflection JSON from plain text, markdown, or wrapped output."""
    try:
        data = loads_json_object(raw_text)
    except JsonObjectParseError as exc:
        raise DiaryReflectionParseError("Diary reflection is not valid JSON") from exc

    try:
        return DiaryReflectionResult.model_validate(data)
    except ValidationError as exc:
        raise DiaryReflectionParseError("Diary reflection does not match schema") from exc
