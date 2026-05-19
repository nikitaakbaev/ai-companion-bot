"""Tool execution for JSON agent decisions."""

import asyncio
import logging
from typing import Any

from aiogram import Bot
from aiogram.types import FSInputFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import AgentActionType, AgentDecision
from app.database.repositories import get_recent_diary_entries
from app.image_generation.service import ImageGenerationService
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
        image_service: ImageGenerationService | None = None,
        max_delay_seconds: float = 0.5,
        typing_seconds: float = 0.2,
    ) -> None:
        self.bot = bot
        self.diary_service = diary_service
        self.image_service = image_service
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
            if decision.action == AgentActionType.REMEMBER:
                return await self._remember(decision, session=session, user_id=user_id)
            if decision.action == AgentActionType.READ_DIARY:
                return await self._read_diary(session=session, user_id=user_id)
            if decision.action == AgentActionType.TAKE_PHOTO:
                return await self._take_photo(decision, telegram_chat_id)
            if decision.action == AgentActionType.UPDATE_IMAGE_BASE_PROMPT:
                return await self._update_image_base_prompt(decision, telegram_chat_id)
            if decision.action == AgentActionType.ANALYZE_IMAGE:
                return ToolExecutionResult(
                    status="unavailable",
                    output={"tool": decision.action.value},
                    error="Tool is not configured yet",
                )

            return ToolExecutionResult(
                status="unsupported",
                output={
                    "tool": decision.action.value,
                    "message": "Unsupported tool action.",
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

    async def _remember(
        self,
        decision: AgentDecision,
        session: AsyncSession | None,
        user_id: int | None,
    ) -> ToolExecutionResult:
        if self.diary_service is None:
            return ToolExecutionResult(status="error", error="Diary service is not configured")
        if session is None or user_id is None:
            return ToolExecutionResult(status="error", error="Remember tool requires session and user_id")

        content = _tool_text(decision, "content", "text", "memory", "value")
        if not content:
            content = "\n".join(decision.normalized_messages())
        if not content:
            return ToolExecutionResult(status="error", error="Remember tool requires content")

        title = _tool_text(decision, "title")
        topics = decision.tool_input.get("topics")
        memory_id = await self.diary_service.remember_manual(
            session=session,
            user_id=user_id,
            content=content,
            title=title,
            topics=topics if isinstance(topics, list) else None,
        )
        return ToolExecutionResult(status="success", output={"memory_id": memory_id})

    async def _read_diary(
        self,
        session: AsyncSession | None,
        user_id: int | None,
    ) -> ToolExecutionResult:
        if session is None or user_id is None:
            return ToolExecutionResult(status="error", error="Read diary tool requires session and user_id")

        entries = await get_recent_diary_entries(session, user_id=user_id, limit=5)
        return ToolExecutionResult(
            status="success",
            output={
                "entries": [
                    {
                        "id": entry.id,
                        "title": entry.title,
                        "summary": entry.summary,
                        "content": entry.content,
                        "importance": entry.importance,
                        "topics": entry.topics or [],
                    }
                    for entry in entries
                ]
            },
        )

    async def _take_photo(
        self,
        decision: AgentDecision,
        telegram_chat_id: int,
    ) -> ToolExecutionResult:
        if self.image_service is None:
            return ToolExecutionResult(status="unavailable", error="Image generation is not configured")

        description = _tool_text(decision, "scene_tags", "description", "prompt", "text")
        mood = _tool_tags_text(
            decision,
            "emotion_tags",
            "environment_tags",
            "lighting_tags",
            "pose_tags",
            "outfit_modifiers",
        ) or _tool_text(decision, "mood")
        style = _tool_tags_text(decision, "camera_tags") or _tool_text(decision, "style")
        negative = _tool_text(decision, "negative", "negative_prompt")
        pre_photo_messages, caption = _photo_messages_and_caption(decision)

        try:
            generation = await self.image_service.generate(
                scene=description,
                mood=mood,
                style=style,
                negative=negative,
            )
        except Exception as exc:
            return ToolExecutionResult(status="error", error=str(exc))

        sent_messages: list[dict[str, Any]] = []
        for text in pre_photo_messages:
            await send_typing(self.bot, telegram_chat_id, seconds=self.typing_seconds)
            message = await self.bot.send_message(chat_id=telegram_chat_id, text=text)
            sent_messages.append({"message_id": message.message_id, "text": text})

        await self.bot.send_chat_action(chat_id=telegram_chat_id, action="upload_photo")
        photo = FSInputFile(generation.image_path)
        message = await self.bot.send_photo(
            chat_id=telegram_chat_id,
            photo=photo,
            caption=caption or None,
        )
        return ToolExecutionResult(
            status="success",
            output={
                "provider": generation.provider,
                "sent_messages": sent_messages,
                "photo_message_id": message.message_id,
                "image_path": generation.image_path,
                "caption": caption,
                "prompt": generation.prompt,
                "negative_prompt": generation.negative_prompt,
                "model": generation.model,
                "preset": generation.preset,
                "telegram_message_id": generation.telegram_message_id,
                "metadata": generation.metadata or {},
            },
        )

    async def _update_image_base_prompt(
        self,
        decision: AgentDecision,
        telegram_chat_id: int,
    ) -> ToolExecutionResult:
        if self.image_service is None:
            return ToolExecutionResult(status="unavailable", error="Image generation is not configured")

        set_tags = _tool_text(decision, "set_tags", "base_tags")
        add_tags = _tool_text(decision, "add_tags", "add")
        remove_tags = _tool_text(decision, "remove_tags", "remove")
        if not any((set_tags, add_tags, remove_tags)):
            return ToolExecutionResult(status="error", error="Base prompt update requires tags")

        output = self.image_service.update_base_prompt(
            add_tags=add_tags,
            remove_tags=remove_tags,
            set_tags=set_tags,
        )
        sent_messages: list[dict[str, Any]] = []
        for text in decision.normalized_messages():
            await send_typing(self.bot, telegram_chat_id, seconds=self.typing_seconds)
            message = await self.bot.send_message(chat_id=telegram_chat_id, text=text)
            sent_messages.append({"message_id": message.message_id, "text": text})
        output["sent_messages"] = sent_messages
        return ToolExecutionResult(status="success", output=output)


def _tool_text(decision: AgentDecision, *keys: str) -> str:
    for key in keys:
        value = decision.tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _tool_tags_text(decision: AgentDecision, *keys: str) -> str:
    tags: list[str] = []
    for key in keys:
        value = decision.tool_input.get(key)
        if isinstance(value, str) and value.strip():
            tags.append(value.strip())
        elif isinstance(value, list):
            tags.extend(str(item).strip() for item in value if str(item).strip())
    return ", ".join(tags)


def _photo_messages_and_caption(decision: AgentDecision) -> tuple[list[str], str]:
    explicit_caption = _tool_text(decision, "caption", "message")
    messages = decision.normalized_messages()
    if explicit_caption:
        return messages, explicit_caption[:1024]
    if not messages:
        return [], ""
    return messages[:-1], messages[-1][:1024]
