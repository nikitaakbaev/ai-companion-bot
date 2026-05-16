"""Tool execution for JSON agent decisions."""

import asyncio
import logging
from typing import Any

from aiogram import Bot
from pydantic import BaseModel, Field

from app.agent.schemas import AgentActionType, AgentDecision
from app.services.typing_service import send_typing

logger = logging.getLogger(__name__)


class ToolExecutionResult(BaseModel):
    """Result of one tool execution."""

    status: str
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ToolExecutor:
    """Executes structured agent actions."""

    def __init__(self, bot: Bot) -> None:
        self.bot = bot

    async def execute(
        self,
        decision: AgentDecision,
        telegram_chat_id: int,
    ) -> ToolExecutionResult:
        """Execute an agent decision."""
        try:
            if decision.action == AgentActionType.SEND_MESSAGE:
                return await self._send_messages(decision, telegram_chat_id)
            if decision.action == AgentActionType.IGNORE:
                return ToolExecutionResult(status="success", output={"ignored": True})

            return ToolExecutionResult(
                status="stub",
                output={
                    "stub": True,
                    "tool": decision.action.value,
                    "message": "Tool is not implemented in stage 4.",
                },
            )
        except Exception as exc:
            logger.exception("Tool execution failed")
            return ToolExecutionResult(status="error", error=str(exc))

    async def _send_messages(
        self,
        decision: AgentDecision,
        telegram_chat_id: int,
    ) -> ToolExecutionResult:
        sent_messages: list[dict[str, Any]] = []
        for text in decision.normalized_messages():
            if decision.delay_seconds > 0:
                await asyncio.sleep(decision.delay_seconds)
            await send_typing(self.bot, telegram_chat_id, seconds=min(max(len(text) / 40, 0.3), 2.0))
            message = await self.bot.send_message(chat_id=telegram_chat_id, text=text)
            sent_messages.append({"message_id": message.message_id, "text": text})

        return ToolExecutionResult(status="success", output={"sent_messages": sent_messages})
