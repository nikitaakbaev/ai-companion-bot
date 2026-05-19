"""Telegram response formatters."""

from app.database.models import AgentAction, BotSettings, DiaryEntry, Message
from app.memory.diary import DiaryServiceResult
from app.memory.rag import RelevantMemory


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
        return "Дневник пока пустой. Напиши несколько сообщений и вызови /sleep."

    lines = ["Дневник:\n"]
    for index, entry in enumerate(entries, start=1):
        topics = ", ".join(entry.topics or []) or "-"
        summary = entry.summary or entry.content
        lines.append(
            f"{index}. {entry.title}\n"
            f"Важность: {entry.importance}/10\n"
            f"Темы: {topics}\n"
            f"Кратко: {summary}"
        )
    return "\n\n".join(lines)


def format_diary_full(entries: list[DiaryEntry]) -> str:
    """Format full diary entries."""
    if not entries:
        return "Дневник пока пустой. Напиши несколько сообщений и вызови /sleep."

    lines = ["Дневник полностью:\n"]
    for index, entry in enumerate(entries, start=1):
        facts_user = "\n".join(f"- {fact}" for fact in (entry.facts_about_user or [])) or "-"
        facts_relationship = (
            "\n".join(f"- {fact}" for fact in (entry.facts_about_relationship or [])) or "-"
        )
        topics = "\n".join(f"- {topic}" for topic in (entry.topics or [])) or "-"
        lines.append(
            f"{index}. {entry.title}\n"
            f"Content: {entry.content}\n"
            "Facts about user:\n"
            f"{facts_user}\n"
            "Facts about relationship:\n"
            f"{facts_relationship}\n"
            "Topics:\n"
            f"{topics}\n"
            f"Emotion: {entry.emotion or '-'}\n"
            f"Importance: {entry.importance}/10"
        )
    return "\n\n".join(lines)


def format_memory(entries: list[DiaryEntry]) -> str:
    """Format memory entries and embedding status."""
    if not entries:
        return "Memory is empty. Create diary entries with /sleep first."

    lines = ["Memory:\n"]
    for index, entry in enumerate(entries, start=1):
        source_date = entry.source_date.isoformat() if entry.source_date else "-"
        embedding = "yes" if entry.embedding_id else "no"
        lines.append(
            f"{index}. {entry.title}\n"
            f"Importance: {entry.importance}/10\n"
            f"Date: {source_date}\n"
            f"Embedding: {embedding}"
        )
    return "\n\n".join(lines)


def format_memory_search(memories: list[RelevantMemory]) -> str:
    """Format memory search results."""
    if not memories:
        return "No relevant memories found."

    lines = ["Found memories:\n"]
    for index, memory in enumerate(memories, start=1):
        text = memory.text.replace("\n", " ").strip()
        if len(text) > 500:
            text = text[:497] + "..."
        lines.append(
            f"{index}. score={memory.score:.2f}\n"
            f"Title: {memory.title or '-'}\n"
            f"Text: {text}"
        )
    return "\n\n".join(lines)


def format_sleep_result(result: DiaryServiceResult) -> str:
    """Format /sleep result."""
    if result.status == "created":
        text = (
            "Я обработал переписку и записал воспоминания в дневник.\n\n"
            f"Создано записей: {result.created_count}"
        )
        if result.day_summary:
            text += f"\n\nКраткая сводка:\n{result.day_summary}"
        if result.indexed_count:
            text += f"\n\nПроиндексировано в память: {result.indexed_count}"
        if result.indexing_failed:
            text += "\nПамять: дневник сохранён, но часть embeddings не создана. Проверь логи."
        return text

    if result.status == "skipped":
        reasons = {
            "already_exists_for_date": "уже есть записи за эту дату.",
            "not_enough_messages": "недостаточно сообщений за период.",
        }
        reason = reasons.get(result.skipped_reason or "", result.skipped_reason or "неизвестно")
        return f"Дневник не создан.\n\nПричина: {reason}"

    if result.status == "empty":
        return "Я просмотрел переписку, но не нашёл ничего достаточно важного для дневника."

    return "Дневник не создан.\n\nПричина: внутренняя ошибка."
