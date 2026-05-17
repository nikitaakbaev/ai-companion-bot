"""LLM orchestration for basic replies and JSON agent decisions."""

import json
import logging
from datetime import UTC, datetime

from pydantic import ValidationError

from app.agent.json_parser import AgentDecisionParseError, parse_agent_decision
from app.agent.prompts import AGENT_RESCUE_PROMPT, AGENT_SYSTEM_PROMPT, BASIC_SYSTEM_PROMPT, JSON_REPAIR_PROMPT
from app.agent.schemas import AgentActionType, AgentDecision, AgentEmotion
from app.agent.tools import available_tools
from app.database.models import AgentState, BotSettings, Message
from app.llm.client import ChatMessage, LLMClient

EMPTY_REPLY_FALLBACK = "Я задумался и не смог нормально сформулировать ответ. Попробуй написать ещё раз."
RESCUE_REPLY_FALLBACK = "Я рядом. Расскажи, что у тебя?"

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Builds a simple chat prompt and asks the LLM for a reply."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_context_messages: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.llm_client = llm_client
        self.max_context_messages = max_context_messages
        self.temperature = temperature
        self.max_tokens = max_tokens

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
            decision = parse_agent_decision(response.content)
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
            json_mode=True,
        )
        logger.info(
            "Received repaired agent response",
            extra={"raw_response_length": len(repair_response.content)},
        )

        try:
            decision = parse_agent_decision(repair_response.content)
            self._log_decision(decision)
            return decision
        except AgentDecisionParseError:
            logger.warning("Failed to parse repaired agent decision; generating plain rescue reply")
            decision = await self._generate_rescue_decision(
                recent_messages=recent_messages,
                event_context=event_context,
                bot_settings=bot_settings,
                agent_state=agent_state,
            )
            self._log_decision(decision)
            return decision

    async def _generate_rescue_decision(
        self,
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
    ) -> AgentDecision:
        messages = self._build_rescue_messages(recent_messages, event_context, bot_settings, agent_state)
        response = await self.llm_client.generate_text(
            messages=messages,
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, 300),
        )
        content = response.content.strip() or RESCUE_REPLY_FALLBACK
        try:
            return AgentDecision(
                thought="Generated a plain user-facing reply after structured JSON failures.",
                action=AgentActionType.SEND_MESSAGE,
                messages=[content],
                tool_input={},
                emotion=AgentEmotion.NEUTRAL,
                delay_seconds=0,
            )
        except ValidationError:
            logger.warning("Plain rescue reply was not user-facing; using neutral rescue fallback")
            return AgentDecision(
                thought="Used neutral rescue fallback after structured JSON failures.",
                action=AgentActionType.SEND_MESSAGE,
                messages=[RESCUE_REPLY_FALLBACK],
                tool_input={},
                emotion=AgentEmotion.NEUTRAL,
                delay_seconds=0,
            )

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
    def _log_decision(decision: AgentDecision) -> None:
        logger.info(
            "Parsed agent decision",
            extra={
                "action": decision.action.value,
                "emotion": decision.emotion.value,
                "message_count": len(decision.normalized_messages()),
            },
        )
