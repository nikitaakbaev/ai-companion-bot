from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentActionType
from app.database.models import BotSettings
from app.llm.client import ChatMessage, LLMClient, LLMResponse


class SequenceLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[ChatMessage]] = []

    async def generate_text(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.responses.pop(0))


def make_orchestrator(llm_client: LLMClient) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_client=llm_client,
        max_context_messages=20,
        temperature=0.7,
        max_tokens=800,
    )


async def test_decide_parses_valid_json() -> None:
    llm = SequenceLLMClient(
        [
            """
            {
              "thought": "ok",
              "action": "send_message",
              "messages": ["Привет"],
              "tool_input": {},
              "emotion": "happy",
              "delay_seconds": 0
            }
            """
        ]
    )

    decision = await make_orchestrator(llm).decide([], {"event_type": "test"})

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Привет"]


async def test_decide_puts_character_settings_into_system_prompt() -> None:
    llm = SequenceLLMClient(
        [
            """
            {
              "thought": "ok",
              "action": "ignore",
              "messages": [],
              "tool_input": {},
              "emotion": "neutral",
              "delay_seconds": 0
            }
            """
        ]
    )
    settings = BotSettings(
        user_id=1,
        character_name="Mira",
        character_description="Always speak like Mira.",
        personality_style="warm and short",
    )

    await make_orchestrator(llm).decide([], {"event_type": "test"}, bot_settings=settings)

    system_prompt = llm.calls[0][0].content
    assert "Имя персонажа: Mira" in system_prompt
    assert "Описание персонажа: Always speak like Mira." in system_prompt
    assert "Стиль общения: warm and short" in system_prompt


async def test_decide_uses_repair_retry() -> None:
    llm = SequenceLLMClient(
        [
            "bad",
            """
            {
              "thought": "fixed",
              "action": "ignore",
              "messages": [],
              "tool_input": {},
              "emotion": "neutral",
              "delay_seconds": 0
            }
            """,
        ]
    )

    decision = await make_orchestrator(llm).decide([], {"event_type": "test"})

    assert decision.action == AgentActionType.IGNORE
    assert len(llm.calls) == 2


async def test_decide_uses_fallback_after_two_parse_failures() -> None:
    llm = SequenceLLMClient(["bad", "still bad"])

    decision = await make_orchestrator(llm).decide([], {"event_type": "test"})

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Я немного запуталась. Напиши ещё раз, пожалуйста."]
