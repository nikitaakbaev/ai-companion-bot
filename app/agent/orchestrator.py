"""Agent orchestration placeholders."""


class AgentOrchestrator:
    """Coordinates event context, LLM decisions, and tool execution."""

    async def handle_event(self) -> None:
        """Handle one incoming event in later stages."""
        raise NotImplementedError

