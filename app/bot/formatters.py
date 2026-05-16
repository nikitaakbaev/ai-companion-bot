"""Telegram response formatters."""

from app.database.models import AgentAction, BotSettings, DiaryEntry, Message


def format_settings(settings: BotSettings) -> str:
    """Format user settings."""
    proactive = "включена" if settings.proactive_enabled else "выключена"
    return (
        "Настройки:\n\n"
        f"Персонаж: {settings.character_name}\n"
        f"Стиль: {settings.personality_style}\n"
        f"LLM model: {settings.llm_model or '-'}\n"
        f"Vision model: {settings.vision_model or '-'}\n"
        f"Embedding model: {settings.embedding_model or '-'}\n\n"
        f"Проактивность: {proactive}\n"
        "Интервал: "
        f"{settings.proactive_min_interval_minutes}-{settings.proactive_max_interval_minutes} минут\n"
        f"Тихие часы: {settings.silent_hours_start}-{settings.silent_hours_end}\n"
        f"Часовой пояс: {settings.timezone}"
    )


def format_history(messages: list[Message]) -> str:
    """Format recent messages."""
    if not messages:
        return "История пока пустая."

    lines = ["Последние сообщения:\n"]
    for message in messages:
        text = (message.text or "").replace("\n", " ").strip()
        if len(text) > 160:
            text = text[:157] + "..."
        lines.append(f"{message.role}: {text}")
    return "\n".join(lines)


def format_actions(actions: list[AgentAction]) -> str:
    """Format recent agent actions."""
    if not actions:
        return "Действий агента пока нет."

    lines = ["Последние действия агента:\n"]
    for action in actions:
        lines.append(f"{action.action_type} — {action.status}")
    return "\n".join(lines)


def format_diary(entries: list[DiaryEntry]) -> str:
    """Format recent diary entries."""
    if not entries:
        return "Дневник пока пустой. Он появится на этапе 6."

    lines = ["Дневник:\n"]
    for index, entry in enumerate(entries, start=1):
        summary = entry.summary or entry.content
        lines.append(f"{index}. {entry.title}\n{summary}")
    return "\n\n".join(lines)
