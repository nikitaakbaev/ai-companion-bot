"""LLM orchestration for basic replies and JSON agent decisions."""

import json
import logging
from datetime import UTC, datetime

from app.agent.json_parser import AgentDecisionParseError, parse_agent_decision
from app.agent.prompts import AGENT_SYSTEM_PROMPT, BASIC_SYSTEM_PROMPT, JSON_REPAIR_PROMPT
from app.agent.schemas import AgentActionType, AgentDecision, AgentEmotion
from app.agent.tools import available_tools
from app.database.models import AgentState, BotSettings, Message
from app.llm.client import ChatMessage, LLMClient
from app.memory.rag import RelevantMemory
from app.memory.relevance import format_memories_for_prompt
from app.memory.profiles import format_profiles_for_prompt

EMPTY_REPLY_FALLBACK = "Я задумался и не смог нормально сформулировать ответ. Попробуй написать ещё раз."
TECHNICAL_HISTORY_PREFIXES = (
    "Сейчас я не могу получить ответ от LLM.",
    "LLM недоступна.",
    "Произошла внутренняя ошибка",
)
FALLBACK_DECISION = AgentDecision(
    thought="LLM returned invalid JSON twice.",
    action=AgentActionType.SEND_MESSAGE,
    messages=["Я немного запуталась. Напиши ещё раз, пожалуйста."],
    tool_input={},
    emotion=AgentEmotion.NEUTRAL,
    delay_seconds=0,
)
AGENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_decision",
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": [
                        "send_message",
                        "ignore",
                        "remember",
                        "read_diary",
                        "sleep",
                        "take_photo",
                        "update_image_base_prompt",
                        "analyze_image",
                    ],
                },
                "messages": {"type": "array", "items": {"type": "string"}},
                "tool_input": {"type": "object"},
                "emotion": {
                    "type": "string",
                    "enum": ["neutral", "happy", "sad", "annoyed", "playful", "caring"],
                },
                "delay_seconds": {"type": "integer", "minimum": 0, "maximum": 60},
            },
            "required": [
                "thought",
                "action",
                "messages",
                "tool_input",
                "emotion",
                "delay_seconds",
            ],
        },
    },
}
MEMORY_PROMPT_RULES = """
Long-term memory rules:
- Use long-term memory only when it is relevant to the current message.
- Do not invent memories that are not present in the memory section.
- If memory conflicts with recent messages, trust recent messages.
- Do not quote memory mechanically; use it naturally in the reply.
""".strip()
IMAGE_PROMPT_RULES = """
Image generation rules:
- take_photo is implemented.
- Use take_photo when the user asks for a photo, selfie, picture, image, appearance, or asks to see you.
- If the user asks to send/show/try a photo anywhere inside a longer multi-topic message, choose take_photo.
- In that case, answer the text part in messages and fulfill the photo request through take_photo.
- You may also use take_photo proactively when a photo would naturally fit the current emotional context.
- Stable Waifu/NekoXL prompt mode is active.
- For take_photo, tool_input must use:
  {"scene_tags":"comma separated anime tags only", "caption":"short natural caption"}.
- Internally think in structured categories: scene, emotion, environment, lighting, camera, pose, outfit modifiers.
- The final tool_input.scene_tags must flatten those categories into one compact comma-separated tag string.
- scene_tags must be lowercase compact anime/booru tags, comma-separated, 15-30 tags max.
- scene_tags must not contain prose, full sentences, markdown, explanations, or camera essays.
- Do not repeat base identity tags: hair color, eye color, base outfit, body, species, core appearance.
- The fixed character identity is already injected through IMAGE_BASE_TAGS by the system.
- Add only scene, pose, emotion, environment, lighting, composition, temporary outfit/accessories.
- Prefer Stable Waifu/NekoXL friendly tags: selfie, mirror selfie, phone camera, close-up, upper body, looking at viewer, holding phone, cozy room, bedroom, gaming setup, rainy window, city lights, blue lighting, soft lighting, neon lighting, screen light.
- Portrait or 9:16 scenes should favor selfie, close-up, upper body, standing pose, phone camera.
- Landscape or 16:9 scenes should favor environment, room, scenery, background-heavy composition.
- Avoid conflicting tags, duplicate tags, random unrelated tags, excessive NSFW tags, prose prompts, Midjourney/Flux-style descriptions, western cartoon, chibi, sketch, comic unless explicitly requested.
- Do not invent base NSFW identity tags. NSFW intensity is controlled by the Python prompt builder configuration.
- For take_photo, write a short natural photo caption in messages[0] or tool_input.caption.
- Good scene_tags: "selfie, cozy room, sleepy, soft lighting, looking at viewer, phone camera".
- Bad scene_tags: "A beautiful anime girl sitting near a window while softly smiling...".
- If the user explicitly asks to permanently change the base visual identity, use update_image_base_prompt instead of repeating base tags in scene_tags.
- update_image_base_prompt tool_input supports {"add_tags":"...", "remove_tags":"...", "set_tags":"..."}.
- Use update_image_base_prompt only for explicit permanent base prompt changes, not for ordinary photos.
""".strip()

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Builds a simple chat prompt and asks the LLM for a reply."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_context_messages: int,
        temperature: float,
        max_tokens: int,
        user_prompt_mode: bool = False,
        response_format_enabled: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.max_context_messages = max_context_messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.user_prompt_mode = user_prompt_mode
        self.response_format_enabled = response_format_enabled

    async def generate_reply(
        self,
        recent_messages: list[Message],
        bot_settings: BotSettings | None = None,
    ) -> str:
        """Generate a Telegram reply from recent persisted messages."""
        system_prompt = BASIC_SYSTEM_PROMPT
        if bot_settings is not None:
            system_prompt = (
                f"{BASIC_SYSTEM_PROMPT}\n\n"
                "Настройки персонажа из .env/BotSettings имеют высокий приоритет.\n"
                f"Имя персонажа: {bot_settings.character_name}\n"
            f"Описание персонажа: {bot_settings.character_description}\n"
            f"Стиль общения: {bot_settings.personality_style}\n"
            "Следуй этим настройкам в обычном ответе пользователю."
            " Если пользователь пишет по-русски, весь ответ должен быть на русском, включая действия в *...*."
        )
        messages = [ChatMessage(role="system", content=system_prompt)]
        seen_history_messages: set[tuple[str, str]] = set()
        for message in recent_messages[-self.max_context_messages :]:
            if message.role not in {"user", "assistant"} or not message.text:
                continue
            if _is_technical_history_message(message.text):
                continue
            history_key = (message.role, _normalize_history_text(message.text))
            if history_key in seen_history_messages:
                continue
            seen_history_messages.add(history_key)
            messages.append(ChatMessage(role=message.role, content=message.text))

        response = await self.llm_client.generate_text(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.content.strip()
        if not content:
            return EMPTY_REPLY_FALLBACK
        return content

    async def decide(
        self,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
        relevant_memories: list[RelevantMemory] | None = None,
    ) -> AgentDecision:
        """Ask the LLM for a structured JSON decision."""
        llm_messages = self._build_agent_messages(
            recent_messages,
            event_context,
            bot_settings,
            agent_state,
            relevant_memories,
        )
        response = await self.llm_client.generate_text(
            messages=llm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=self._agent_response_format(),
        )
        logger.info("Received raw agent response", extra={"raw_response_length": len(response.content)})

        try:
            decision = parse_agent_decision(response.content)
            decision = _coerce_photo_request_decision(decision, event_context)
            self._log_decision(decision)
            return decision
        except AgentDecisionParseError:
            logger.warning("Failed to parse agent decision; attempting JSON repair")

        repair_response = await self.llm_client.generate_text(
            messages=[
                *llm_messages,
                ChatMessage(
                    role="user",
                    content=JSON_REPAIR_PROMPT.replace("{raw_text}", response.content[:4000]),
                ),
            ],
            temperature=0,
            max_tokens=self.max_tokens,
            response_format=self._agent_response_format(),
        )
        logger.info(
            "Received repaired agent response",
            extra={"raw_response_length": len(repair_response.content)},
        )

        try:
            decision = parse_agent_decision(repair_response.content)
            decision = _coerce_photo_request_decision(decision, event_context)
            self._log_decision(decision)
            return decision
        except AgentDecisionParseError:
            logger.warning("Failed to parse repaired agent decision; using fallback")
            return FALLBACK_DECISION.model_copy(deep=True)

    def _build_agent_messages(
        self,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
        relevant_memories: list[RelevantMemory] | None = None,
    ) -> list[ChatMessage]:
        compact_tools = [
            {"name": tool["name"], "implemented": tool["implemented"]} for tool in available_tools()
        ]
        enriched_context = {
            **event_context,
            "current_time": datetime.now(UTC).isoformat(),
            "available_tools": compact_tools,
        }
        if bot_settings is not None:
            enriched_context["character"] = {
                "name": bot_settings.character_name,
                "description": bot_settings.character_description,
                "personality_style": bot_settings.personality_style,
            }
        if agent_state is not None:
            enriched_context["agent_state"] = {
                "last_emotion": agent_state.last_emotion,
                "last_action_type": agent_state.last_action_type,
            }
        profiles = event_context.get("profiles")
        if isinstance(profiles, dict):
            user_profile = dict(profiles.get("user") or {})
            character_profile = dict(profiles.get("character") or {})
            enriched_context["profiles"] = {
                "user": user_profile,
                "character": character_profile,
            }
            enriched_context["structured_profiles"] = format_profiles_for_prompt(
                user_profile,
                character_profile,
            )
        system_prompt = self._build_system_prompt(bot_settings)
        if self.user_prompt_mode:
            messages = [
                ChatMessage(
                    role="user",
                    content=system_prompt + "\n\nИстория и текущий Event context будут ниже.",
                )
            ]
        else:
            messages = [ChatMessage(role="system", content=system_prompt)]
        seen_history_messages: set[tuple[str, str]] = set()
        for message in recent_messages[-self.max_context_messages :]:
            if message.role not in {"user", "assistant"} or not message.text:
                continue
            if _is_technical_history_message(message.text):
                continue
            history_key = (message.role, _normalize_history_text(message.text))
            if history_key in seen_history_messages:
                continue
            seen_history_messages.add(history_key)
            messages.append(ChatMessage(role=message.role, content=message.text))

        messages.append(
            ChatMessage(
                role="user",
                content=self._format_event_context(
                    event_context,
                    enriched_context,
                    relevant_memories or [],
                ),
            )
        )
        return messages

    @staticmethod
    def _build_system_prompt(bot_settings: BotSettings | None = None) -> str:
        if bot_settings is None:
            return f"{AGENT_SYSTEM_PROMPT}\n\n{MEMORY_PROMPT_RULES}\n\n{IMAGE_PROMPT_RULES}"

        return (
            f"{AGENT_SYSTEM_PROMPT}\n\n{MEMORY_PROMPT_RULES}\n\n{IMAGE_PROMPT_RULES}\n\n"
            "Настройки персонажа из .env/BotSettings имеют высокий приоритет.\n"
            f"Имя персонажа: {bot_settings.character_name}\n"
            f"Описание персонажа: {bot_settings.character_description}\n"
            f"Стиль общения: {bot_settings.personality_style}\n"
            "Следуй этим настройкам при выборе сообщений в JSON.\n"
            "Если пользователь пишет по-русски, поле messages должно быть целиком на русском, включая действия в *...*.\n"
            "В поле messages пиши только обычную реплику пользователю, без упоминаний JSON, схемы, формата или внутренней логики."
        )

    @staticmethod
    def _log_decision(decision: AgentDecision) -> None:
        logger.info(
            "Parsed agent decision",
            extra={
                "action": decision.action.value,
                "emotion": decision.emotion.value,
                "message_count": len(decision.normalized_messages()),
            },
        )

    def _agent_response_format(self) -> dict | None:
        if not self.response_format_enabled:
            return None
        return AGENT_RESPONSE_FORMAT

    @staticmethod
    def _format_event_context(
        event_context: dict,
        enriched_context: dict,
        relevant_memories: list[RelevantMemory],
    ) -> str:
        user_text = event_context.get("text")
        sections: list[str] = []
        if isinstance(user_text, str) and user_text.strip():
            sections.append(
                "Current user message to answer:\n"
                f"{user_text.strip()}"
            )
        sections.append("Structured profiles:\n" + str(enriched_context.get("structured_profiles", "")))
        sections.append("Long-term memory:\n" + format_memories_for_prompt(relevant_memories))
        sections.append(
            "Event context (service data, do not repeat it to the user):\n"
            + json.dumps(enriched_context, ensure_ascii=False, separators=(",", ":"))
        )
        return "\n\n".join(sections)

        prefix = ""
        if isinstance(user_text, str) and user_text.strip():
            prefix = (
                "Последнее сообщение пользователя, на которое нужно ответить:\n"
                f"{user_text.strip()}\n\n"
            )
        return (
            prefix
            + "Event context (служебные данные, не пересказывай их пользователю):\n"
            + json.dumps(enriched_context, ensure_ascii=False, separators=(",", ":"))
        )


def _is_technical_history_message(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    if normalized.startswith("/"):
        return True
    readable_prefixes = (
        "Сейчас я не могу получить ответ от LLM.",
        "LLM недоступна.",
        "Произошла внутренняя ошибка",
        "Я немного запуталась.",
        "Stable Waifu test image.",
    )
    return any(normalized.startswith(prefix) for prefix in TECHNICAL_HISTORY_PREFIXES) or any(
        normalized.startswith(prefix) for prefix in readable_prefixes
    )


def _normalize_history_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _coerce_photo_request_decision(
    decision: AgentDecision,
    event_context: dict,
) -> AgentDecision:
    if decision.action == AgentActionType.TAKE_PHOTO:
        return decision
    if not event_context.get("photo_request_detected"):
        return decision

    coerced = decision.model_copy(deep=True)
    coerced.action = AgentActionType.TAKE_PHOTO
    coerced.tool_input = {
        **coerced.tool_input,
        "scene_tags": coerced.tool_input.get("scene_tags")
        or _scene_tags_from_photo_request(str(event_context.get("text") or "")),
    }
    if not coerced.messages:
        coerced.messages = ["Сейчас попробую."]
    logger.info("Coerced agent decision to take_photo for explicit photo request")
    return coerced


def _scene_tags_from_photo_request(text: str) -> str:
    lowered = text.casefold()
    tags = ["selfie", "phone camera", "looking at viewer", "soft lighting"]
    if any(word in lowered for word in ("кровать", "bed", "леж", "lying")):
        tags.extend(["bedroom", "lying on bed", "cozy room"])
    elif any(word in lowered for word in ("нож", "ног", "feet", "legs")):
        tags.extend(["sitting", "full body", "cozy room"])
    elif any(word in lowered for word in ("ноутбук", "laptop", "компьютер", "gaming")):
        tags.extend(["desk", "laptop", "screen light", "cozy room"])
    else:
        tags.extend(["cozy room", "upper body"])
    return ", ".join(tags)
