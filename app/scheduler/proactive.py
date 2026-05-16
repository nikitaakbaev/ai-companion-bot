"""Proactive message scheduling placeholders."""


class ProactiveScheduler:
    """Schedules proactive companion events."""

    async def run_once(self) -> None:
        """Run one proactive cycle in later stages."""
        raise NotImplementedError

