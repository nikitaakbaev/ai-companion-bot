"""Diary summarization placeholders."""


class DiarySummarizer:
    """Summarizes conversations into diary entries."""

    async def summarize(self) -> list[dict]:
        """Summarize messages in later stages."""
        raise NotImplementedError

