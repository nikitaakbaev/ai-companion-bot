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

EMPTY_REPLY_FALLBACK = "Я задумался и не смог нормально сформулировать ответ. Попробуй написать ещё раз."
FALLBACK_DECISION = AgentDecision(
    thought="LLM returned invalid JSON twice.",
    action=AgentActionType.SEND_MESSAGE,
    messages=["Я немного запутался с форматом ответа. Напиши ещё раз."],
    tool_input={},
    emotion=AgentEmotion.NEUTRAL,
    delay_seconds=0,
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
                ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=JSON_REPAIR_PROMPT.replace("{raw_text}", response.content[:4000]),
                ),
            ],
            temperature=0,
            max_tokens=self.max_tokens,
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
            logger.warning("Failed to parse repaired agent decision; using fallback")
            return FALLBACK_DECISION.model_copy(deep=True)

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
        messages = [ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT)]
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
    def _log_decision(decision: AgentDecision) -> None:
        logger.info(
            "Parsed agent decision",
            extra={
                "action": decision.action.value,
                "emotion": decision.emotion.value,
                "message_count": len(decision.normalized_messages()),
            },
        )
