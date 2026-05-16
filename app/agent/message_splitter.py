"""Utilities for splitting long text into Telegram-sized messages."""

import re


def split_into_messages(text: str, max_length: int = 500) -> list[str]:
    """Split text into non-empty chunks without losing content."""
    normalized = text.strip()
    if not normalized:
        return []
    if max_length <= 0:
        raise ValueError("max_length must be greater than zero")
    if len(normalized) <= max_length:
        return [normalized]

    parts = [part.strip() for part in re.split(r"(\n{2,}|(?<=[.!?])\s+)", normalized)]
    chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = ""

    for part in parts:
        if not part or part.isspace():
            continue
        if len(part) > max_length:
            flush_current()
            chunks.extend(part[i : i + max_length] for i in range(0, len(part), max_length))
            continue
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= max_length:
            current = candidate
        else:
            flush_current()
            current = part

    flush_current()
    return [chunk for chunk in chunks if chunk]

