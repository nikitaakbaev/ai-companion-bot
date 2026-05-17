from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentActionType
from app.database.models import BotSettings
from app.llm.client import ChatMessage, LLMClient, LLMResponse


class SequenceLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
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
    assert llm.json_modes == [True]
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
    assert llm.json_modes == [True, True]


async def test_decide_repairs_service_leak_in_messages_with_context() -> None:
    llm = SequenceLLMClient(
        [
            """
            {
              "thought": "leaked service text",
              "action": "send_message",
              "messages": ["Спасибо, что исправил! Теперь я знаю, как надо!~"],
              "tool_input": {},
              "emotion": "happy",
              "delay_seconds": 0
            }
            """,
            """
            {
              "thought": "fixed user-facing reply",
              "action": "send_message",
              "messages": ["Да, я с тобой."],
              "tool_input": {},
              "emotion": "caring",
              "delay_seconds": 0
            }
            """,
        ]
    )

    decision = await make_orchestrator(llm).decide([], {"event_type": "test", "text": "Ты со мной?"})

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Да, я с тобой."]
    assert len(llm.calls) == 2
    assert "Event context" in llm.calls[1][-2].content
    assert "Спасибо, что исправил" in llm.calls[1][-1].content


async def test_decide_generates_plain_rescue_reply_after_two_parse_failures() -> None:
    llm = SequenceLLMClient(["bad", "still bad", "Привет-привет. Я тут."])

    decision = await make_orchestrator(llm).decide([], {"event_type": "test"})

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Привет-привет. Я тут."]
    assert llm.json_modes == [True, True, False]


async def test_decide_uses_neutral_rescue_fallback_when_plain_rescue_is_service_text() -> None:
    llm = SequenceLLMClient(["bad", "still bad", "Here is the corrected valid JSON response"])

    decision = await make_orchestrator(llm).decide([], {"event_type": "test"})

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Я рядом. Расскажи, что у тебя?"]
    assert llm.json_modes == [True, True, False]
