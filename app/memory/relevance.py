"""Formatting helpers for relevant long-term memories."""

from app.memory.rag import RelevantMemory


def format_memories_for_prompt(
    memories: list[RelevantMemory],
    max_chars: int | None = None,
) -> str:
    """Format memories for the agent prompt."""
    if not memories:
        return "No relevant long-term memories found."

    lines = ["Relevant long-term memories:"]
    used_chars = len(lines[0])
    for index, memory in enumerate(memories, start=1):
        title = memory.title or "Untitled memory"
        text = memory.text.strip()
        block = f"\n{index}. [score={memory.score:.2f}] {title}\n{text}"
        if max_chars is not None and used_chars + len(block) > max_chars:
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            block = block[:remaining]
        lines.append(block)
        used_chars += len(block)
    return "\n".join(lines)


def is_relevant(score: float, threshold: float) -> bool:
    """Return whether a similarity score passes the configured threshold."""
    return score >= threshold
