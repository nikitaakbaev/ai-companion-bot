"""Diary summarization through LLM reflection."""

import logging
from datetime import date

from app.database.models import Message
from app.llm.client import ChatMessage, LLMClient
from app.memory.json_parser import DiaryReflectionParseError, parse_diary_reflection
from app.memory.schemas import DiaryReflectionResult

logger = logging.getLogger(__name__)

DIARY_REFLECTION_PROMPT = """
Ты модуль рефлексии AI companion.

Твоя задача — проанализировать переписку за период и создать дневниковые записи памяти.

Не пересказывай каждое сообщение.
Выделяй только факты, предпочтения, планы, эмоциональные моменты, темы, которые могут быть полезны в будущем.

Не выдумывай факты.
Если данных мало — верни пустой список entries.

Верни только JSON без markdown.

Формат:
{
  "day_summary": "краткая сводка периода",
  "entries": [
    {
      "title": "короткий заголовок",
      "content": "подробная запись памяти",
      "summary": "короткая версия",
      "facts_about_user": ["..."],
      "facts_about_relationship": ["..."],
      "topics": ["..."],
      "importance": 1,
      "emotion": "neutral",
      "source_date": "YYYY-MM-DD"
    }
  ]
}

Правила:
- source_date должна быть датой анализируемого периода.
- importance от 1 до 10.
- Не больше указанного максимума записей.
- Не включать приватные токены, ключи, пароли.
- Не сохранять случайный мусор.
- Сохранять только то, что реально поможет будущему диалогу.
""".strip()

DIARY_REPAIR_PROMPT = """
Ты вернул невалидный JSON для дневника. Исправь ответ и верни только валидный JSON:
{
  "day_summary": "string | null",
  "entries": [
    {
      "title": "string",
      "content": "string",
      "summary": "string | null",
      "facts_about_user": ["string"],
      "facts_about_relationship": ["string"],
      "topics": ["string"],
      "importance": 5,
      "emotion": "neutral",
      "source_date": "YYYY-MM-DD"
    }
  ]
}

Невалидный ответ:
{raw_text}
""".strip()


class DiarySummarizer:
    """Summarizes conversation history into diary entries."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_entries_per_run: int,
        max_input_chars: int,
    ) -> None:
        self.llm_client = llm_client
        self.max_entries_per_run = max_entries_per_run
        self.max_input_chars = max_input_chars

    async def summarize(
        self,
        messages: list[Message],
        source_date: date,
    ) -> DiaryReflectionResult:
        """Summarize messages into structured diary entries."""
        if not messages:
            return DiaryReflectionResult(entries=[], day_summary=None)

        history_text = self._format_messages(messages)
        truncated_history = history_text[: self.max_input_chars]
        logger.info(
            "Sending diary reflection request",
            extra={"message_count": len(messages), "input_chars": len(truncated_history)},
        )
        user_prompt = (
            f"Дата периода: {source_date.isoformat()}\n"
            f"Максимум записей: {self.max_entries_per_run}\n\n"
            "История переписки:\n"
            f"{truncated_history}"
        )
        response = await self.llm_client.generate_text(
            messages=[
                ChatMessage(role="system", content=DIARY_REFLECTION_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        result = await self._parse_or_repair(response.content)
        result.entries = result.entries[: self.max_entries_per_run]
        logger.info("Diary reflection parsed", extra={"entry_count": len(result.entries)})
        return result

    async def _parse_or_repair(self, raw_text: str) -> DiaryReflectionResult:
        try:
            return parse_diary_reflection(raw_text)
        except DiaryReflectionParseError:
            logger.warning("Failed to parse diary reflection; attempting repair")

        repair_response = await self.llm_client.generate_text(
            messages=[
                ChatMessage(role="system", content=DIARY_REFLECTION_PROMPT),
                ChatMessage(
                    role="user",
                    content=DIARY_REPAIR_PROMPT.replace("{raw_text}", raw_text[:4000]),
                ),
            ],
            temperature=0,
            max_tokens=1200,
        )
        try:
            return parse_diary_reflection(repair_response.content)
        except DiaryReflectionParseError:
            logger.exception("Failed to parse repaired diary reflection")
            return DiaryReflectionResult(entries=[], day_summary=None)

    @staticmethod
    def _format_messages(messages: list[Message]) -> str:
        lines: list[str] = []
        for message in messages:
            created_at = message.created_at.strftime("%Y-%m-%d %H:%M")
            text = (message.text or "").replace("\n", " ").strip()
            lines.append(f"[{created_at}] {message.role}: {text}")
        return "\n".join(lines)
