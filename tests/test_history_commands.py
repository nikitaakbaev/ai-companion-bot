from app.bot.formatters import (
    format_actions,
    format_diary,
    format_diary_full,
    format_history,
    format_settings,
    format_sleep_result,
)
from app.database.models import AgentAction, BotSettings, DiaryEntry, Message
from app.memory.diary import DiaryServiceResult


def test_format_history_empty() -> None:
    assert format_history([]) == "История пока пустая."


def test_format_history_messages() -> None:
    text = format_history(
        [
            Message(role="user", text="Привет", message_type="text", chat_id=1),
            Message(role="assistant", text="Я на связи", message_type="text", chat_id=1),
        ]
    )

    assert "user: Привет" in text
    assert "assistant: Я на связи" in text


def test_format_actions() -> None:
    text = format_actions([AgentAction(action_type="send_message", status="success")])

    assert "send_message — success" in text


def test_format_settings() -> None:
    settings = BotSettings(
        user_id=1,
        character_name="Kuni",
        character_description="desc",
        personality_style="warm",
        llm_model="qwen",
        vision_model="qwen-vl",
        embedding_model="embed",
        proactive_enabled=False,
        proactive_min_interval_minutes=60,
        proactive_max_interval_minutes=180,
        timezone="Europe/Moscow",
        silent_hours_start="23:00",
        silent_hours_end="09:00",
    )

    text = format_settings(settings)

    assert "Персонаж: Kuni" in text
    assert "Проактивность: выключена" in text


def test_format_diary_empty_and_entries() -> None:
    assert format_diary([]) == "Дневник пока пустой. Напиши несколько сообщений и вызови /sleep."

    text = format_diary(
        [
            DiaryEntry(
                user_id=1,
                title="Day",
                content="Content",
                summary="Summary",
                topics=["LLM"],
                importance=7,
            )
        ]
    )

    assert "1. Day" in text
    assert "Важность: 7/10" in text
    assert "Темы: LLM" in text
    assert "Кратко: Summary" in text


def test_format_diary_full() -> None:
    text = format_diary_full(
        [
            DiaryEntry(
                user_id=1,
                title="Day",
                content="Content",
                facts_about_user=["uses Ollama"],
                facts_about_relationship=["building bot"],
                topics=["LLM"],
                emotion="happy",
                importance=8,
            )
        ]
    )

    assert "Дневник полностью:" in text
    assert "Content: Content" in text
    assert "- uses Ollama" in text
    assert "Importance: 8/10" in text


def test_format_sleep_result() -> None:
    created = format_sleep_result(
        DiaryServiceResult(status="created", created_count=2, day_summary="Summary")
    )
    skipped = format_sleep_result(
        DiaryServiceResult(status="skipped", skipped_reason="not_enough_messages")
    )
    empty = format_sleep_result(DiaryServiceResult(status="empty"))

    assert "Создано записей: 2" in created
    assert "недостаточно сообщений" in skipped
    assert "не нашёл ничего достаточно важного" in empty
