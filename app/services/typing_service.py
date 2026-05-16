"""Typing indicator service placeholders."""


class TypingService:
    """Controls typing indicators."""

    async def show_typing(self, chat_id: int) -> None:
        """Show typing in later stages."""
        raise NotImplementedError

