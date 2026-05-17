"""JSON decision parsing for the agent loop."""

from pydantic import ValidationError

from app.agent.schemas import AgentDecision
from app.utils.json import JsonObjectParseError, loads_json_object


class AgentDecisionParseError(Exception):
    """Raised when an LLM response cannot be parsed as an agent decision."""


def parse_agent_decision(raw_text: str) -> AgentDecision:
    """Parse an AgentDecision from plain JSON, markdown JSON, or text around JSON."""
    try:
        data = loads_json_object(raw_text)
    except JsonObjectParseError as exc:
        raise AgentDecisionParseError("Agent decision is not valid JSON") from exc

    try:
        return AgentDecision.model_validate(data)
    except ValidationError as exc:
        raise AgentDecisionParseError("Agent decision does not match schema") from exc
