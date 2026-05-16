"""Tool execution placeholders."""

from app.agent.schemas import AgentDecision


class ToolExecutor:
    """Executes structured agent actions."""

    async def execute(self, decision: AgentDecision) -> None:
        """Execute an agent decision in later stages."""
        raise NotImplementedError

