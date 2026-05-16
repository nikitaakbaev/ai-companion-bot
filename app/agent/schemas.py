"""Agent schemas."""

from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    """Structured action returned by the future LLM agent loop."""

    thought: str = ""
    action: str = "ignore"
    messages: list[str] = Field(default_factory=list)
    tool_input: dict = Field(default_factory=dict)
    emotion: str = "neutral"
    delay_seconds: int = 0

