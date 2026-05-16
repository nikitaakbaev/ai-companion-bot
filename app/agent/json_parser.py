"""JSON decision parsing for the agent loop."""

import json
import re

from pydantic import ValidationError

from app.agent.schemas import AgentDecision


class AgentDecisionParseError(Exception):
    """Raised when an LLM response cannot be parsed as an agent decision."""


def parse_agent_decision(raw_text: str) -> AgentDecision:
    """Parse an AgentDecision from plain JSON, markdown JSON, or text around JSON."""
    candidate = _extract_json(raw_text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AgentDecisionParseError("Agent decision is not valid JSON") from exc

    try:
        return AgentDecision.model_validate(data)
    except ValidationError as exc:
        raise AgentDecisionParseError("Agent decision does not match schema") from exc


def _extract_json(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        raise AgentDecisionParseError("Agent decision is empty")

    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    raise AgentDecisionParseError("Agent decision JSON object was not found")
