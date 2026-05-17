from datetime import date

import pytest
from pydantic import ValidationError

from app.memory.json_parser import DiaryReflectionParseError, parse_diary_reflection
from app.memory.schemas import DiaryEntryCreate


VALID_JSON = """
{
  "day_summary": "Обсудили локальные LLM.",
  "entries": [
    {
      "title": "Локальные LLM",
      "content": "Пользователь настраивал Ollama.",
      "summary": "Настройка Ollama.",
      "facts_about_user": ["использует Ollama"],
      "facts_about_relationship": [],
      "topics": ["LLM", ""],
      "importance": 7,
      "emotion": "curious",
      "source_date": "2026-05-16"
    }
  ]
}
""".strip()


def test_parse_clean_diary_json() -> None:
    result = parse_diary_reflection(VALID_JSON)

    assert result.day_summary == "Обсудили локальные LLM."
    assert len(result.entries) == 1
    assert result.entries[0].topics == ["LLM"]


def test_parse_diary_json_from_markdown() -> None:
    result = parse_diary_reflection(f"```json\n{VALID_JSON}\n```")

    assert result.entries[0].title == "Локальные LLM"


def test_parse_diary_json_with_text_around() -> None:
    result = parse_diary_reflection(f"before\n{VALID_JSON}\nafter")

    assert result.entries[0].source_date.isoformat() == "2026-05-16"


def test_parse_diary_json_with_trailing_commas() -> None:
    raw = """
    {
      "day_summary": "Summary",
      "entries": [
        {
          "title": "Entry",
          "content": "Content",
          "source_date": "2026-05-16",
        },
      ],
    }
    """

    result = parse_diary_reflection(raw)

    assert result.entries[0].title == "Entry"


def test_invalid_diary_json_raises() -> None:
    with pytest.raises(DiaryReflectionParseError):
        parse_diary_reflection("not json")


def test_diary_importance_validation() -> None:
    with pytest.raises(ValidationError):
        DiaryEntryCreate(
            title="x",
            content="x",
            importance=11,
            source_date=date(2026, 5, 16),
        )
