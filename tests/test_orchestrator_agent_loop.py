from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentActionType
from app.database.models import BotSettings
from app.llm.client import ChatMessage, LLMClient, LLMResponse


class SequenceLLMClient(LLMClient):
    def __init__(self, responses: list[str | LLMResponse]) -> None:
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
        response = self.responses.pop(0)
        if isinstance(response, LLMResponse):
            return response
        return LLMResponse(content=response)


class SequenceResponseVerifier:
    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts = verdicts
        self.candidate_messages: list[list[str]] = []

    async def is_sendable(self, candidate_messages: list[str], **kwargs) -> bool:
        self.candidate_messages.append(candidate_messages)
        return self.verdicts.pop(0)


def make_orchestrator(
    llm_client: LLMClient,
    response_verifier: SequenceResponseVerifier | None = None,
) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_client=llm_client,
        max_context_messages=20,
        temperature=0.7,
        max_tokens=800,
        response_verifier=response_verifier,
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


async def test_decide_plain_reply_uses_plain_text_without_json_mode() -> None:
    llm = SequenceLLMClient(["Привет-привет."])

    decision = await make_orchestrator(llm).decide_plain_reply(
        [],
        {"event_type": "telegram_text_message", "text": "Привет"},
    )

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Привет-привет."]
    assert llm.json_modes == [False]


async def test_decide_plain_reply_retries_unsuitable_plain_response() -> None:
    llm = SequenceLLMClient(["Я рядом. Расскажи, что у тебя?", "Привет. Я слушаю."])

    decision = await make_orchestrator(llm).decide_plain_reply(
        [],
        {"event_type": "telegram_text_message", "text": "Привет"},
    )

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Привет. Я слушаю."]
    assert llm.json_modes == [False, False]
    assert "rejected_replies" in llm.calls[-1][-1].content


async def test_decide_plain_reply_suppresses_when_no_response_is_sendable() -> None:
    llm = SequenceLLMClient(["Here is the corrected valid JSON response", "Вот исправленный JSON"])

    decision = await make_orchestrator(llm).decide_plain_reply(
        [],
        {"event_type": "telegram_text_message", "text": "Привет"},
    )

    assert decision.action == AgentActionType.IGNORE
    assert decision.normalized_messages() == []
    assert llm.json_modes == [False, False]


async def test_decide_plain_reply_continues_when_finish_reason_is_length() -> None:
    llm = SequenceLLMClient(
        [
            LLMResponse(content="У меня отлично. Ты заряжаешь меня своим", finish_reason="length"),
            LLMResponse(content="теплом.", finish_reason="stop"),
        ]
    )

    decision = await make_orchestrator(llm).decide_plain_reply(
        [],
        {"event_type": "telegram_text_message", "text": "Как дела?"},
    )

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["У меня отлично. Ты заряжаешь меня своим теплом."]
    assert llm.json_modes == [False, False]
    assert "partial_response" in llm.calls[1][-1].content


async def test_decide_plain_reply_continues_long_sentence_without_terminal_punctuation() -> None:
    partial = "У меня всё хорошо, и я улыбаюсь, потому что твои сообщения делают этот день теплее своим"
    llm = SequenceLLMClient([partial, "настроением."])

    decision = await make_orchestrator(llm).decide_plain_reply(
        [],
        {"event_type": "telegram_text_message", "text": "Как ты?"},
    )

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == [f"{partial} настроением."]
    assert llm.json_modes == [False, False]


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


async def test_decide_repairs_when_verifier_rejects_valid_json_response() -> None:
    llm = SequenceLLMClient(
        [
            """
            {
              "thought": "valid but unrelated",
              "action": "send_message",
              "messages": ["О, я заметила твою ошибку."],
              "tool_input": {},
              "emotion": "happy",
              "delay_seconds": 0
            }
            """,
            """
            {
              "thought": "fixed relevant reply",
              "action": "send_message",
              "messages": ["Привет-привет."],
              "tool_input": {},
              "emotion": "happy",
              "delay_seconds": 0
            }
            """,
        ]
    )
    verifier = SequenceResponseVerifier([False, True])

    decision = await make_orchestrator(llm, response_verifier=verifier).decide(
        [],
        {"event_type": "test", "text": "Привет"},
    )

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Привет-привет."]
    assert verifier.candidate_messages == [["О, я заметила твою ошибку."], ["Привет-привет."]]
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


async def test_decide_retries_generic_plain_rescue_reply() -> None:
    llm = SequenceLLMClient(
        [
            "bad",
            "still bad",
            "Я рядом. Расскажи, что у тебя?",
            "Привет. Я тебя слышу.",
        ]
    )

    decision = await make_orchestrator(llm).decide([], {"event_type": "test", "text": "Привет"})

    assert decision.action == AgentActionType.SEND_MESSAGE
    assert decision.normalized_messages() == ["Привет. Я тебя слышу."]
    assert llm.json_modes == [True, True, False, False]
    assert "rejected_replies" in llm.calls[-1][-1].content


async def test_decide_suppresses_response_when_plain_rescue_is_not_user_facing() -> None:
    llm = SequenceLLMClient(
        [
            "bad",
            "still bad",
            "Here is the corrected valid JSON response",
            "Вот исправленный валидный JSON",
        ]
    )

    decision = await make_orchestrator(llm).decide([], {"event_type": "test"})

    assert decision.action == AgentActionType.IGNORE
    assert decision.normalized_messages() == []
    assert llm.json_modes == [True, True, False, False]
    assert "rejected_replies" in llm.calls[-1][-1].content
