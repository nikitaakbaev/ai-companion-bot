"""Lightweight user-facing response verifier."""

import json
import logging

from pydantic import BaseModel, Field, ValidationError

from app.database.models import AgentState, BotSettings, Message
from app.llm.client import ChatMessage, LLMClient, LLMClientError
from app.utils.json import JsonObjectParseError, loads_json_object

logger = logging.getLogger(__name__)

RESPONSE_VERIFIER_PROMPT = """
Ты быстрый проверяющий перед отправкой ответа в Telegram.

Твоя задача: решить, можно ли отправить candidate_response пользователю.

Верни только JSON:
{
  "is_sendable": true,
  "reason": "short reason"
}

Ставь is_sendable=false, если:
- ответ не связан с последним сообщением пользователя;
- ответ выглядит как системная заглушка, repair, fallback или техническое сообщение;
- ответ упоминает JSON, схему, формат, ошибку, исправление, внутреннюю обработку;
- ответ просит "напиши ещё раз" без реальной причины;
- ответ выдумывает событие, которого нет в контексте;
- ответ игнорирует простой смысл последнего сообщения пользователя.

Ставь is_sendable=true, если:
- ответ естественно продолжает диалог;
- ответ логически подходит к последнему сообщению пользователя;
- ответ звучит как обычная реплика персонажа в Telegram.
""".strip()


class ResponseVerificationResult(BaseModel):
    """Structured verdict returned by the verifier model."""

    is_sendable: bool = Field(default=False)
    reason: str = Field(default="")


class AgentResponseVerifier:
    """Checks candidate assistant messages before they are sent to Telegram."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_context_messages: int,
        max_tokens: int = 200,
    ) -> None:
        self.llm_client = llm_client
        self.max_context_messages = max_context_messages
        self.max_tokens = max_tokens

    async def is_sendable(
        self,
        candidate_messages: list[str],
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None = None,
        agent_state: AgentState | None = None,
    ) -> bool:
        """Return whether candidate messages are safe and relevant to send."""
        if not candidate_messages:
            return False

        payload = self._build_payload(
            candidate_messages=candidate_messages,
            recent_messages=recent_messages,
            event_context=event_context,
            bot_settings=bot_settings,
            agent_state=agent_state,
        )
        try:
            response = await self.llm_client.generate_text(
                messages=[
                    ChatMessage(role="system", content=RESPONSE_VERIFIER_PROMPT),
                    ChatMessage(
                        role="user",
                        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    ),
                ],
                temperature=0,
                max_tokens=self.max_tokens,
                json_mode=True,
            )
            data = loads_json_object(response.content)
            verdict = ResponseVerificationResult.model_validate(data)
        except (LLMClientError, JsonObjectParseError, ValidationError):
            logger.exception("Response verifier failed; rejecting candidate response")
            return False

        if not verdict.is_sendable:
            logger.info("Response verifier rejected candidate", extra={"reason": verdict.reason})
        return verdict.is_sendable

    def _build_payload(
        self,
        candidate_messages: list[str],
        recent_messages: list[Message],
        event_context: dict,
        bot_settings: BotSettings | None,
        agent_state: AgentState | None,
    ) -> dict:
        history = [
            {"role": message.role, "text": message.text}
            for message in recent_messages[-self.max_context_messages :]
            if message.role in {"user", "assistant"} and message.text
        ]
        payload = {
            "event_context": event_context,
            "recent_messages": history,
            "candidate_response": candidate_messages,
        }
        if bot_settings is not None:
            payload["character"] = {
                "name": bot_settings.character_name,
                "description": bot_settings.character_description,
                "personality_style": bot_settings.personality_style,
            }
        if agent_state is not None:
            payload["agent_state"] = {
                "last_emotion": agent_state.last_emotion,
                "last_action_type": agent_state.last_action_type,
            }
        return payload
