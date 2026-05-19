from datetime import date

from app.bot.formatters import format_memory, format_memory_search
from app.database.models import DiaryEntry
from app.memory.rag import RelevantMemory


def test_format_memory_shows_embedding_status() -> None:
    text = format_memory(
        [
            DiaryEntry(
                title="Memory",
                content="Content",
                importance=7,
                source_date=date(2026, 5, 17),
                embedding_id="diary_entry:1",
            )
        ]
    )

    assert "Memory" in text
    assert "Embedding: yes" in text


def test_format_memory_search_results() -> None:
    text = format_memory_search(
        [RelevantMemory(score=0.82, title="Title", text="Text", diary_entry_id=1)]
    )

    assert "score=0.82" in text
    assert "Title: Title" in text
