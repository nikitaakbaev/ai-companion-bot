"""Simple stage 3 LLM orchestration."""

from app.agent.prompts import BASIC_SYSTEM_PROMPT
from app.database.models import Message
from app.llm.client import ChatMessage, LLMClient

EMPTY_REPLY_FALLBACK = "Я задумался и не смог нормально сформулировать ответ. Попробуй написать ещё раз."


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
