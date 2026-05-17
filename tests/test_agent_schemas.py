import pytest
from pydantic import ValidationError

from app.agent.schemas import AgentActionType, AgentDecision, SERVICE_LEAK_FALLBACK_MESSAGE


def test_valid_send_message_decision() -> None:
    decision = AgentDecision(
        action="send_message",
        messages=["Привет"],
        emotion="happy",
        delay_seconds=1,
    )

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Привет"]


def test_send_message_requires_messages() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(action="send_message", messages=[])


def test_ignore_allows_empty_messages() -> None:
    decision = AgentDecision(action="ignore", messages=[])

    assert decision.action == AgentActionType.IGNORE
    assert decision.normalized_messages() == []


def test_empty_messages_are_removed() -> None:
    decision = AgentDecision(action="send_message", messages=[" ", "ok", ""])

    assert decision.normalized_messages() == ["ok"]


def test_long_message_is_truncated() -> None:
    decision = AgentDecision(action="send_message", messages=["x" * 2500])

    assert len(decision.normalized_messages()[0]) == 2000


def test_common_unknown_emotion_alias_is_normalized() -> None:
    decision = AgentDecision(action="send_message", messages=["ok"], emotion="helpful")

    assert decision.emotion == "caring"


def test_common_action_alias_is_normalized() -> None:
    decision = AgentDecision(action="reply", messages="ok", tool_input=None, delay_seconds=0.4)

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["ok"]
    assert decision.tool_input == {}
    assert decision.delay_seconds == 0


def test_service_json_leak_message_is_replaced() -> None:
    decision = AgentDecision(
        action="send_message",
        messages=["Here is the corrected valid JSON schema response"],
    )

    assert decision.normalized_messages() == [SERVICE_LEAK_FALLBACK_MESSAGE]


def test_russian_service_json_leak_message_is_replaced() -> None:
    decision = AgentDecision(
        action="send_message",
        messages=["Вот исправленный валидный JSON, как вы просили"],
    )

    assert decision.normalized_messages() == [SERVICE_LEAK_FALLBACK_MESSAGE]
