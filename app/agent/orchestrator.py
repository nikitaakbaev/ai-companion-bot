"""LLM orchestration for basic replies and JSON agent decisions."""

import json
import logging
import re
from datetime import UTC, datetime

from pydantic import ValidationError

from app.agent.json_parser import AgentDecisionParseError, parse_agent_decision
from app.agent.prompts import (
    AGENT_RESCUE_PROMPT,
    AGENT_SYSTEM_PROMPT,
    BASIC_SYSTEM_PROMPT,
    JSON_REPAIR_PROMPT,
    PLAIN_CONTINUATION_PROMPT,
    PLAIN_CHAT_PROMPT,
)
from app.agent.response_verifier import AgentResponseVerifier
from app.agent.schemas import AgentActionType, AgentDecision, AgentEmotion
from app.agent.tools import available_tools
from app.database.models import AgentState, BotSettings, Message
from app.llm.client import ChatMessage, LLMClient

EMPTY_REPLY_FALLBACK = "Я задумался и не смог нормально сформулировать ответ. Попробуй написать ещё раз."
MAX_RESCUE_REPLY_ATTEMPTS = 2
MAX_PLAIN_CONTINUATION_ATTEMPTS = 2
INCOMPLETE_REPLY_MIN_LENGTH = 80
TERMINAL_REPLY_CHARS = frozenset(".!?…。！？)")
UNSUITABLE_RESCUE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"я\s+немного\s+запут",
        r"напиши\s+е[щш]е?\s+раз",
        r"попробуй\s+написать\s+е[щш]е?\s+раз",
        r"я\s+рядом[.!]?\s+расскажи",
        r"расскажи,\s+что\s+у\s+тебя",
        r"не\s+смог\w*\s+сформулировать",
        r"не\s+могу\s+ответить",
    )
)

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Builds a simple chat prompt and asks the LLM for a reply."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_context_messages: int,
        temperature: float,
        max_tokens: int,
        response_verifier: AgentResponseVerifier | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_context_messages = max_context_messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_verifier = response_verifier

    async def generate_reply(
        self,
        recent_messages: list[Message],
    ) -> str:
        """Generate a Telegram reply from recent persisted messages."""
        messages = [ChatMessage(role="system", content=BASIC_SYSTEM_PROMPT)]
        for message in recent_messages[-self.max_context_messages :]:
            if message.role not in {"user", "assistant"} or not message.text:
                continue
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
    ) -> AgentDecision:
        """Ask the LLM for a structured JSON decision."""
        llm_messages = self._build_agent_messages(
            recent_messages,
            event_context,
            bot_settings,
            agent_state,
        )
        response = await self.llm_client.generate_text(
            messages=llm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_mode=True,
        )
        logger.info("Received raw agent response", extra={"raw_response_length": len(response.content)})

        try:
            decision = await self._parse_verified_decision(
                response.content,
                recent_messages=recent_messages,
                event_context=event_context,
                bot_settings=bot_settings,
                agent_state=agent_state,
            )
            self._log_decision(decision)
            return decision
        except AgentDecisionParseError:
            logger.warning("Failed to parse or verify agent decision; attempting JSON repair")

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
            json_mode=True,
        )
        logger.info(
            "Received repaired agent response",
            extra={"raw_response_length": len(repair_response.content)},
        )

        try:
            decision = await self._parse_verified_decision(
                repair_response.content,
                recent_messages=recent_messages,
                event_context=event_context,
                bot_settings=bot_settings,
                agent_state=agent_state,
            )
            self._log_decision(decision)
            return decision
        except AgentDecisionParseError:
            logger.warning("Failed to parse or verify repaired agent decision; generating plain rescue reply")
            decision = await self._generate_rescue_decision(
                recent_messages=recent_messages,
                event_context=event_context,
                bot_settings=bot_settings,
                agent_state=agent_state,
            )
            self._log_decision(decision)
            return decision

    async def decide_plain_reply(
        self,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
    ) -> AgentDecision:
        """Generate a normal text reply for ordinary chat without JSON agent loop."""
        decision = await self._generate_plain_decision(
            recent_messages=recent_messages,
            event_context=event_context,
            bot_settings=bot_settings,
            agent_state=agent_state,
        )
        self._log_decision(decision)
        return decision

    async def _generate_plain_decision(
        self,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
    ) -> AgentDecision:
        rejected_replies: list[str] = []
        for _ in range(MAX_RESCUE_REPLY_ATTEMPTS):
            messages = self._build_plain_messages(
                recent_messages,
                event_context,
                bot_settings,
                agent_state,
                rejected_replies=rejected_replies,
            )
            response = await self.llm_client.generate_text(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = await self._complete_plain_response_if_needed(
                content=response.content.strip(),
                response_finish_reason=response.finish_reason,
                messages=messages,
            )
            if not content:
                rejected_replies.append("<empty>")
                continue
            try:
                decision = AgentDecision(
                    thought="Generated plain chat reply.",
                    action=AgentActionType.SEND_MESSAGE,
                    messages=[content],
                    tool_input={},
                    emotion=AgentEmotion.NEUTRAL,
                    delay_seconds=0,
                )
                if await self._is_plain_decision_sendable(
                    decision,
                    recent_messages=recent_messages,
                    event_context=event_context,
                    bot_settings=bot_settings,
                    agent_state=agent_state,
                ):
                    return decision
                logger.warning("Plain chat reply looked unsuitable; retrying")
                rejected_replies.append(content[:500])
            except ValidationError:
                logger.warning("Plain chat reply was not user-facing; retrying")
                rejected_replies.append(content[:500])

        logger.warning("Failed to produce a user-facing plain reply; suppressing response")
        return AgentDecision(
            thought="Suppressed response because no plain user-facing reply passed validation.",
            action=AgentActionType.IGNORE,
            messages=[],
            tool_input={},
            emotion=AgentEmotion.NEUTRAL,
            delay_seconds=0,
        )

    async def _complete_plain_response_if_needed(
        self,
        content: str,
        response_finish_reason: str | None,
        messages: list[ChatMessage],
    ) -> str:
        completed = content.strip()
        for _ in range(MAX_PLAIN_CONTINUATION_ATTEMPTS):
            if not self._looks_incomplete_reply(completed, response_finish_reason):
                break

            logger.warning("Plain reply looks incomplete; requesting continuation")
            continuation_response = await self.llm_client.generate_text(
                messages=[
                    *messages,
                    ChatMessage(role="assistant", content=completed),
                    ChatMessage(
                        role="user",
                        content=PLAIN_CONTINUATION_PROMPT.replace(
                            "{partial_response}",
                            completed[-2000:],
                        ),
                    ),
                ],
                temperature=0,
                max_tokens=min(self.max_tokens, 500),
            )
            continuation = continuation_response.content.strip()
            if not continuation:
                break
            completed = self._join_continuation(completed, continuation)
            response_finish_reason = continuation_response.finish_reason
        return completed.strip()

    async def _generate_rescue_decision(
        self,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
    ) -> AgentDecision:
        rejected_replies: list[str] = []
        for _ in range(MAX_RESCUE_REPLY_ATTEMPTS):
            messages = self._build_rescue_messages(
                recent_messages,
                event_context,
                bot_settings,
                agent_state,
                rejected_replies=rejected_replies,
            )
            response = await self.llm_client.generate_text(
                messages=messages,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, 300),
            )
            content = response.content.strip()
            if not content:
                rejected_replies.append("<empty>")
                continue
            try:
                decision = AgentDecision(
                    thought="Generated a plain user-facing reply after structured JSON failures.",
                    action=AgentActionType.SEND_MESSAGE,
                    messages=[content],
                    tool_input={},
                    emotion=AgentEmotion.NEUTRAL,
                    delay_seconds=0,
                )
                if await self._is_rescue_decision_sendable(
                    decision,
                    recent_messages=recent_messages,
                    event_context=event_context,
                    bot_settings=bot_settings,
                    agent_state=agent_state,
                ):
                    return decision
                logger.warning("Plain rescue reply looked generic; retrying")
                rejected_replies.append(content[:500])
            except ValidationError:
                logger.warning("Plain rescue reply was not user-facing; retrying")
                rejected_replies.append(content[:500])

        logger.warning("Failed to produce a user-facing rescue reply; suppressing response")
        return AgentDecision(
            thought="Suppressed response because no user-facing reply passed validation.",
            action=AgentActionType.IGNORE,
            messages=[],
            tool_input={},
            emotion=AgentEmotion.NEUTRAL,
            delay_seconds=0,
        )

    async def _parse_verified_decision(
        self,
        raw_text: str,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
    ) -> AgentDecision:
        decision = parse_agent_decision(raw_text)
        if await self._is_decision_sendable(
            decision,
            recent_messages=recent_messages,
            event_context=event_context,
            bot_settings=bot_settings,
            agent_state=agent_state,
        ):
            return decision
        raise AgentDecisionParseError("Agent decision was rejected by response verifier")

    def _build_agent_messages(
        self,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
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
        messages = [ChatMessage(role="system", content=self._build_system_prompt(bot_settings))]
        for message in recent_messages[-self.max_context_messages :]:
            if message.role not in {"user", "assistant"} or not message.text:
                continue
            messages.append(ChatMessage(role=message.role, content=message.text))

        messages.append(
            ChatMessage(
                role="user",
                content="Event context:\n"
                + json.dumps(enriched_context, ensure_ascii=False, separators=(",", ":")),
            )
        )
        return messages

    def _build_rescue_messages(
        self,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
        rejected_replies: list[str] | None = None,
    ) -> list[ChatMessage]:
        enriched_context = {
            **event_context,
            "current_time": datetime.now(UTC).isoformat(),
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
        if rejected_replies:
            enriched_context["rejected_replies"] = rejected_replies

        messages = [ChatMessage(role="system", content=self._build_rescue_system_prompt(bot_settings))]
        for message in recent_messages[-self.max_context_messages :]:
            if message.role not in {"user", "assistant"} or not message.text:
                continue
            messages.append(ChatMessage(role=message.role, content=message.text))
        messages.append(
            ChatMessage(
                role="user",
                content="Event context:\n"
                + json.dumps(enriched_context, ensure_ascii=False, separators=(",", ":")),
            )
        )
        return messages

    def _build_plain_messages(
        self,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
        rejected_replies: list[str] | None = None,
    ) -> list[ChatMessage]:
        enriched_context = {
            **event_context,
            "current_time": datetime.now(UTC).isoformat(),
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
        if rejected_replies:
            enriched_context["rejected_replies"] = rejected_replies

        messages = [ChatMessage(role="system", content=self._build_plain_system_prompt(bot_settings))]
        for message in recent_messages[-self.max_context_messages :]:
            if message.role not in {"user", "assistant"} or not message.text:
                continue
            messages.append(ChatMessage(role=message.role, content=message.text))
        messages.append(
            ChatMessage(
                role="user",
                content="Event context:\n"
                + json.dumps(enriched_context, ensure_ascii=False, separators=(",", ":")),
            )
        )
        return messages

    @staticmethod
    def _build_system_prompt(bot_settings: BotSettings | None = None) -> str:
        if bot_settings is None:
            return AGENT_SYSTEM_PROMPT

        return (
            f"{AGENT_SYSTEM_PROMPT}\n\n"
            "Настройки персонажа из .env/BotSettings имеют высокий приоритет.\n"
            f"Имя персонажа: {bot_settings.character_name}\n"
            f"Описание персонажа: {bot_settings.character_description}\n"
            f"Стиль общения: {bot_settings.personality_style}\n"
            "Следуй этим настройкам при выборе сообщений в JSON.\n"
            "В поле messages пиши только обычную реплику пользователю, без упоминаний JSON, схемы, формата или внутренней логики."
        )

    @staticmethod
    def _build_rescue_system_prompt(bot_settings: BotSettings | None = None) -> str:
        if bot_settings is None:
            return AGENT_RESCUE_PROMPT

        return (
            f"{AGENT_RESCUE_PROMPT}\n\n"
            "Настройки персонажа из .env/BotSettings имеют высокий приоритет.\n"
            f"Имя персонажа: {bot_settings.character_name}\n"
            f"Описание персонажа: {bot_settings.character_description}\n"
            f"Стиль общения: {bot_settings.personality_style}\n"
            "Следуй этим настройкам в обычном ответе пользователю."
        )

    @staticmethod
    def _build_plain_system_prompt(bot_settings: BotSettings | None = None) -> str:
        if bot_settings is None:
            return PLAIN_CHAT_PROMPT

        return (
            f"{PLAIN_CHAT_PROMPT}\n\n"
            "Настройки персонажа из .env/BotSettings имеют высокий приоритет.\n"
            f"Имя персонажа: {bot_settings.character_name}\n"
            f"Описание персонажа: {bot_settings.character_description}\n"
            f"Стиль общения: {bot_settings.personality_style}\n"
            "Следуй этим настройкам в обычном ответе пользователю."
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

    async def _is_decision_sendable(
        self,
        decision: AgentDecision,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
    ) -> bool:
        if decision.action != AgentActionType.SEND_MESSAGE:
            return True
        messages = decision.normalized_messages()
        if not messages:
            return False
        if self.response_verifier is None:
            return True
        return await self.response_verifier.is_sendable(
            candidate_messages=messages,
            recent_messages=recent_messages,
            event_context=event_context,
            bot_settings=bot_settings,
            agent_state=agent_state,
        )

    async def _is_rescue_decision_sendable(
        self,
        decision: AgentDecision,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
    ) -> bool:
        if decision.action != AgentActionType.SEND_MESSAGE:
            return False
        messages = decision.normalized_messages()
        if not messages:
            return False
        combined_text = "\n".join(messages)
        if any(pattern.search(combined_text) for pattern in UNSUITABLE_RESCUE_PATTERNS):
            return False
        return await self._is_decision_sendable(
            decision,
            recent_messages=recent_messages,
            event_context=event_context,
            bot_settings=bot_settings,
            agent_state=agent_state,
        )

    async def _is_plain_decision_sendable(
        self,
        decision: AgentDecision,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
    ) -> bool:
        if decision.action != AgentActionType.SEND_MESSAGE:
            return False
        messages = decision.normalized_messages()
        if not messages:
            return False
        combined_text = "\n".join(messages)
        if any(pattern.search(combined_text) for pattern in UNSUITABLE_RESCUE_PATTERNS):
            return False
        return await self._is_decision_sendable(
            decision,
            recent_messages=recent_messages,
            event_context=event_context,
            bot_settings=bot_settings,
            agent_state=agent_state,
        )

    @staticmethod
    def _looks_incomplete_reply(content: str, finish_reason: str | None) -> bool:
        text = content.strip()
        if not text:
            return False
        if finish_reason == "length":
            return True
        if len(text) < INCOMPLETE_REPLY_MIN_LENGTH:
            return False
        stripped = text.rstrip("*_`~ \t\r\n\"'»）)]}")
        if not stripped:
            return False
        return stripped[-1] not in TERMINAL_REPLY_CHARS

    @staticmethod
    def _join_continuation(prefix: str, continuation: str) -> str:
        left = prefix.rstrip()
        right = continuation.strip()
        if not left:
            return right
        if not right:
            return left
        if right.startswith((".", ",", "!", "?", ":", ";", "…")):
            return f"{left}{right}"
        return f"{left} {right}"
