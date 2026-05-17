import pytest

from app.agent.json_parser import AgentDecisionParseError, parse_agent_decision
from app.agent.schemas import AgentActionType


VALID_JSON = """
{
  "thought": "ok",
  "action": "send_message",
  "messages": ["Привет"],
  "tool_input": {},
  "emotion": "happy",
  "delay_seconds": 1
}
""".strip()


def test_parse_clean_json() -> None:
    decision = parse_agent_decision(VALID_JSON)

    assert decision.action == AgentActionType.SEND_MESSAGE


def test_parse_json_from_markdown() -> None:
    decision = parse_agent_decision(f"```json\n{VALID_JSON}\n```")

    assert decision.normalized_messages() == ["Привет"]


def test_parse_json_with_text_around_it() -> None:
    decision = parse_agent_decision(f"before\n{VALID_JSON}\nafter")

    assert decision.emotion == "happy"


def test_parse_first_balanced_json_object() -> None:
    raw = f"before\n{VALID_JSON}\nafter {{not json}}"

    decision = parse_agent_decision(raw)

    assert decision.action == AgentActionType.SEND_MESSAGE


def test_parse_json_with_trailing_commas() -> None:
    raw = """
    {
      "thought": "ok",
      "action": "send_message",
      "messages": ["Hi",],
      "tool_input": {},
      "emotion": "happy",
      "delay_seconds": 1,
    }
    """

    decision = parse_agent_decision(raw)

    assert decision.normalized_messages() == ["Hi"]


def test_parse_python_style_dict() -> None:
    raw = "{'thought':'ok','action':'send_message','messages':['Hi'],'tool_input':{},'emotion':'happy','delay_seconds':1}"

    decision = parse_agent_decision(raw)

    assert decision.normalized_messages() == ["Hi"]


def test_invalid_json_raises_parse_error() -> None:
    with pytest.raises(AgentDecisionParseError):
        parse_agent_decision("not json")
