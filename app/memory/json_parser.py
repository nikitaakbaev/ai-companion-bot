"""Diary reflection JSON parser."""

import json
import re

from pydantic import ValidationError

from app.memory.schemas import DiaryReflectionResult


class DiaryReflectionParseError(Exception):
    """Raised when diary reflection JSON cannot be parsed."""


def parse_diary_reflection(raw_text: str) -> DiaryReflectionResult:
    """Parse diary reflection JSON from plain text, markdown, or wrapped output."""
    candidate = _extract_json(raw_text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise DiaryReflectionParseError("Diary reflection is not valid JSON") from exc

    try:
        return DiaryReflectionResult.model_validate(data)
    except ValidationError as exc:
        raise DiaryReflectionParseError("Diary reflection does not match schema") from exc


def _extract_json(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        raise DiaryReflectionParseError("Diary reflection is empty")

    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    raise DiaryReflectionParseError("Diary reflection JSON object was not found")
