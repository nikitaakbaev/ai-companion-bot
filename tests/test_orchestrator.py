from app.agent.orchestrator import EMPTY_REPLY_FALLBACK, AgentOrchestrator
from app.agent.prompts import BASIC_SYSTEM_PROMPT
from app.database.models import Message
from app.llm.client import ChatMessage, LLMClient, LLMResponse


class FakeLLMClient(LLMClient):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[ChatMessage]] = []

    async def generate_text(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.content)


async def test_orchestrator_adds_system_prompt_and_history_in_order() -> None:
    llm = FakeLLMClient("reply")
    orchestrator = AgentOrchestrator(
        llm_client=llm,
        max_context_messages=20,
        temperature=0.7,
        max_tokens=800,
    )
    history = [
        Message(role="user", text="hello", message_type="text", chat_id=1),
        Message(role="assistant", text="hi", message_type="text", chat_id=1),
    ]

    reply = await orchestrator.generate_reply(history)

    assert reply == "reply"
    assert llm.calls[0] == [
        ChatMessage(role="system", content=BASIC_SYSTEM_PROMPT),
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi"),
    ]


async def test_orchestrator_returns_fallback_for_empty_reply() -> None:
    llm = FakeLLMClient("   ")
    orchestrator = AgentOrchestrator(
        llm_client=llm,
        max_context_messages=20,
        temperature=0.7,
        max_tokens=800,
    )

    reply = await orchestrator.generate_reply([])

    assert reply == EMPTY_REPLY_FALLBACK
