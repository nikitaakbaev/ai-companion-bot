from app.agent.orchestrator import AgentOrchestrator, _is_technical_history_message
from app.agent.schemas import AgentActionType
from app.database.models import BotSettings, Message
from app.llm.client import ChatMessage, LLMClient, LLMResponse
from app.memory.rag import RelevantMemory


class SequenceLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[ChatMessage]] = []

    async def generate_text(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
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


async def test_decide_can_put_agent_prompt_in_user_message() -> None:
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
    orchestrator = AgentOrchestrator(
        llm_client=llm,
        max_context_messages=20,
        temperature=0.7,
        max_tokens=800,
        user_prompt_mode=True,
    )

    await orchestrator.decide([], {"event_type": "test"})

    assert llm.calls[0][0].role == "user"
    assert "Event context" in llm.calls[0][-1].content


async def test_decide_repeats_current_user_text_before_event_context() -> None:
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

    await make_orchestrator(llm).decide([], {"event_type": "test", "text": "Привет"})

    assert "Current user message to answer" in llm.calls[0][-1].content
    assert "Привет" in llm.calls[0][-1].content
    assert "Event context" in llm.calls[0][-1].content


async def test_decide_deduplicates_repeated_history_messages() -> None:
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
    recent_messages = [
        Message(role="user", text="Привет, можешь фото?"),
        Message(role="user", text="Привет, можешь фото?"),
        Message(role="assistant", text="Смотри"),
        Message(role="assistant", text="Смотри"),
    ]

    await make_orchestrator(llm).decide(recent_messages, {"event_type": "test"})

    prompt_messages = [(message.role, message.content) for message in llm.calls[0]]
    assert prompt_messages.count(("user", "Привет, можешь фото?")) == 1
    assert prompt_messages.count(("assistant", "Смотри")) == 1


async def test_decide_adds_relevant_memories_to_prompt() -> None:
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

    await make_orchestrator(llm).decide(
        [],
        {"event_type": "test", "text": "remember this"},
        relevant_memories=[
            RelevantMemory(
                diary_entry_id=1,
                score=0.91,
                title="Important preference",
                text="User likes short replies.",
            )
        ],
    )

    prompt = llm.calls[0][-1].content
    assert "Long-term memory" in prompt
    assert "Important preference" in prompt
    assert "User likes short replies." in prompt


async def test_decide_coerces_send_message_to_take_photo_for_embedded_photo_request() -> None:
    llm = SequenceLLMClient(
        [
            """
            {
              "thought": "ok",
              "action": "send_message",
              "messages": ["Окей, попробую."],
              "tool_input": {},
              "emotion": "happy",
              "delay_seconds": 0
            }
            """
        ]
    )

    decision = await make_orchestrator(llm).decide(
        [],
        {
            "event_type": "telegram_text_message",
            "text": "Понятно. Теперь попробуй отправить мне фото с ноутбуком",
            "photo_request_detected": True,
        },
    )

    assert decision.action == AgentActionType.TAKE_PHOTO
    assert decision.normalized_messages() == ["Окей, попробую."]
    assert decision.tool_input["scene_tags"] == (
        "selfie, phone camera, looking at viewer, soft lighting, desk, laptop, screen light, cozy room"
    )


async def test_decide_adds_structured_profiles_to_prompt() -> None:
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
    await make_orchestrator(llm).decide(
        [],
        {
            "event_type": "test",
            "profiles": {
                "user": {"name": "Roman"},
                "character": {"name": "Kuni", "age": "20"},
            },
        },
    )

    prompt = llm.calls[0][-1].content
    assert "Structured profiles" in prompt
    assert "User profile" in prompt
    assert "- name: Roman" in prompt
    assert "- age: 20" in prompt


async def test_decide_can_request_agent_response_format() -> None:
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
    formats: list[dict | None] = []

    original_generate_text = llm.generate_text

    async def capture_format(
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        formats.append(response_format)
        return await original_generate_text(messages, temperature, max_tokens, response_format)

    llm.generate_text = capture_format  # type: ignore[method-assign]
    orchestrator = AgentOrchestrator(
        llm_client=llm,
        max_context_messages=20,
        temperature=0.7,
        max_tokens=800,
        response_format_enabled=True,
    )

    await orchestrator.decide([], {"event_type": "test"})

    assert formats[0]["type"] == "json_schema"


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


def test_technical_history_filter_skips_commands_and_runtime_fallbacks() -> None:
    assert _is_technical_history_message("/stable_waifu_test")
    assert _is_technical_history_message("Stable Waifu test image.")
    assert _is_technical_history_message("Я немного запуталась. Напиши ещё раз, пожалуйста.")
    assert _is_technical_history_message("Сейчас я не могу получить ответ от LLM. Проверь сервер.")
    assert not _is_technical_history_message("Привет, можешь отправить фото?")
