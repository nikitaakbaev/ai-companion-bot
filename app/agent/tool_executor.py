"""Tool execution for JSON agent decisions."""

import asyncio
import logging
from typing import Any

from aiogram import Bot
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import AgentActionType, AgentDecision
from app.memory.diary import DiaryService
from app.services.typing_service import send_typing

logger = logging.getLogger(__name__)


class ToolExecutionResult(BaseModel):
    """Result of one tool execution."""

    status: str
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ToolExecutor:
    """Executes structured agent actions."""

    def __init__(
        self,
        bot: Bot,
        diary_service: DiaryService | None = None,
        max_delay_seconds: float = 0.5,
        typing_seconds: float = 0.2,
    ) -> None:
        self.bot = bot
        self.diary_service = diary_service
        self.max_delay_seconds = max_delay_seconds
        self.typing_seconds = typing_seconds

    async def execute(
        self,
        decision: AgentDecision,
        telegram_chat_id: int,
        session: AsyncSession | None = None,
        user_id: int | None = None,
    ) -> ToolExecutionResult:
        """Execute an agent decision."""
        try:
            if decision.action == AgentActionType.SEND_MESSAGE:
                return await self._send_messages(decision, telegram_chat_id)
            if decision.action == AgentActionType.IGNORE:
                return ToolExecutionResult(status="success", output={"ignored": True})
            if decision.action == AgentActionType.SLEEP:
                return await self._sleep(session=session, user_id=user_id)

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
                await asyncio.sleep(min(decision.delay_seconds, self.max_delay_seconds))
            await send_typing(self.bot, telegram_chat_id, seconds=self.typing_seconds)
            message = await self.bot.send_message(chat_id=telegram_chat_id, text=text)
            sent_messages.append({"message_id": message.message_id, "text": text})

        return ToolExecutionResult(status="success", output={"sent_messages": sent_messages})

    async def _sleep(
        self,
        session: AsyncSession | None,
        user_id: int | None,
    ) -> ToolExecutionResult:
        if self.diary_service is None:
            return ToolExecutionResult(status="error", error="Diary service is not configured")
        if session is None or user_id is None:
            return ToolExecutionResult(status="error", error="Sleep tool requires session and user_id")

        result = await self.diary_service.create_daily_summary(session=session, user_id=user_id)
        return ToolExecutionResult(status=result.status, output=result.model_dump(mode="json"))
