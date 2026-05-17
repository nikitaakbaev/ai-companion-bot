from datetime import UTC, date, datetime

from app.database.models import Message
from app.llm.client import ChatMessage, LLMClient, LLMResponse
from app.memory.summarizer import DiarySummarizer


class SequenceLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[ChatMessage]] = []
        self.json_modes: list[bool] = []

    async def generate_text(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(messages)
        self.json_modes.append(json_mode)
        return LLMResponse(content=self.responses.pop(0))


def make_message(role: str, text: str) -> Message:
    return Message(
        role=role,
        text=text,
        message_type="text",
        chat_id=1,
        created_at=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
    )


def diary_json(title: str = "Entry") -> str:
    return f"""
    {{
      "day_summary": "Summary",
      "entries": [
        {{
          "title": "{title}",
          "content": "Content",
          "summary": "Short",
          "facts_about_user": ["fact"],
          "facts_about_relationship": [],
          "topics": ["topic"],
          "importance": 6,
          "emotion": "neutral",
          "source_date": "2026-05-16"
        }}
      ]
    }}
    """


async def test_summarizer_returns_reflection_result() -> None:
    llm = SequenceLLMClient([diary_json()])
    summarizer = DiarySummarizer(llm, max_entries_per_run=8, max_input_chars=20000)

    result = await summarizer.summarize([make_message("user", "Привет")], date(2026, 5, 16))

    assert result.day_summary == "Summary"
    assert result.entries[0].title == "Entry"
    assert llm.json_modes == [True]


async def test_summarizer_limits_entries() -> None:
    raw = """
    {
      "day_summary": "Summary",
      "entries": [
        {"title":"One","content":"C","source_date":"2026-05-16"},
        {"title":"Two","content":"C","source_date":"2026-05-16"}
      ]
    }
    """
    summarizer = DiarySummarizer(SequenceLLMClient([raw]), max_entries_per_run=1, max_input_chars=20000)

    result = await summarizer.summarize([make_message("user", "x")], date(2026, 5, 16))

    assert len(result.entries) == 1
    assert result.entries[0].title == "One"


async def test_summarizer_empty_messages_returns_empty() -> None:
    summarizer = DiarySummarizer(SequenceLLMClient([]), max_entries_per_run=8, max_input_chars=20000)

    result = await summarizer.summarize([], date(2026, 5, 16))

    assert result.entries == []


async def test_summarizer_repairs_invalid_json() -> None:
    llm = SequenceLLMClient(["bad", diary_json("Fixed")])
    summarizer = DiarySummarizer(llm, max_entries_per_run=8, max_input_chars=20000)

    result = await summarizer.summarize([make_message("user", "x")], date(2026, 5, 16))

    assert result.entries[0].title == "Fixed"
    assert len(llm.calls) == 2
    assert llm.json_modes == [True, True]


async def test_summarizer_returns_empty_after_two_invalid_jsons() -> None:
    llm = SequenceLLMClient(["bad", "still bad"])
    summarizer = DiarySummarizer(llm, max_entries_per_run=8, max_input_chars=20000)

    result = await summarizer.summarize([make_message("user", "x")], date(2026, 5, 16))

    assert result.entries == []
