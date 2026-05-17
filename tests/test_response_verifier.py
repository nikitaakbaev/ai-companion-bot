from app.agent.response_verifier import AgentResponseVerifier
from app.database.models import Message
from app.llm.client import ChatMessage, LLMClient, LLMResponse


class FakeVerifierLLMClient(LLMClient):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[ChatMessage]] = []
        self.json_modes: list[bool] = []

    async def generate_text(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(messages)
        self.json_modes.append(json_mode)
        return LLMResponse(content=self.content)


async def test_response_verifier_accepts_sendable_response() -> None:
    llm = FakeVerifierLLMClient('{"is_sendable": true, "reason": "relevant"}')
    verifier = AgentResponseVerifier(llm_client=llm, max_context_messages=8)

    is_sendable = await verifier.is_sendable(
        candidate_messages=["Привет-привет."],
        recent_messages=[Message(role="user", text="Привет", message_type="text", chat_id=1)],
        event_context={"event_type": "test", "text": "Привет"},
    )

    assert is_sendable is True
    assert llm.json_modes == [True]
    assert "candidate_response" in llm.calls[0][1].content


async def test_response_verifier_rejects_unsendable_response() -> None:
    llm = FakeVerifierLLMClient('{"is_sendable": false, "reason": "unrelated"}')
    verifier = AgentResponseVerifier(llm_client=llm, max_context_messages=8)

    is_sendable = await verifier.is_sendable(
        candidate_messages=["Я заметила твою ошибку."],
        recent_messages=[Message(role="user", text="Привет", message_type="text", chat_id=1)],
        event_context={"event_type": "test", "text": "Привет"},
    )

    assert is_sendable is False


async def test_response_verifier_fails_closed_on_bad_verdict() -> None:
    verifier = AgentResponseVerifier(
        llm_client=FakeVerifierLLMClient("not json"),
        max_context_messages=8,
    )

    is_sendable = await verifier.is_sendable(
        candidate_messages=["Привет."],
        recent_messages=[],
        event_context={"event_type": "test"},
    )

    assert is_sendable is False
