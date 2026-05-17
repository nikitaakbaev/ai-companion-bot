"""Agent schemas."""

from enum import StrEnum
import re

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_AGENT_MESSAGE_LENGTH = 2000
SERVICE_LEAK_FALLBACK_MESSAGE = "Я тебя поняла. Можешь чуть уточнить, что именно ты хочешь?"
SERVICE_LEAK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bjson\b",
        r"\bvalid\b",
        r"\bschema\b",
        r"\btool\b",
        r"\baction\b",
        r"валидн",
        r"схем",
        r"формат",
        r"исправленн",
        r"служебн",
    )
)


class AgentActionType(StrEnum):
    """Actions supported by the JSON agent loop."""

    SEND_MESSAGE = "send_message"
    IGNORE = "ignore"
    REMEMBER = "remember"
    READ_DIARY = "read_diary"
    SLEEP = "sleep"
    TAKE_PHOTO = "take_photo"
    ANALYZE_IMAGE = "analyze_image"


class AgentEmotion(StrEnum):
    """Allowed companion emotion labels."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANNOYED = "annoyed"
    PLAYFUL = "playful"
    CARING = "caring"


class AgentDecision(BaseModel):
    """Structured action returned by the LLM agent loop."""

    thought: str = Field(default="")
    action: AgentActionType
    messages: list[str] = Field(default_factory=list)
    tool_input: dict = Field(default_factory=dict)
    emotion: AgentEmotion = AgentEmotion.NEUTRAL
    delay_seconds: int = Field(default=0, ge=0, le=60)

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value: object) -> AgentActionType | str:
        """Map common model-invented action labels to the supported set."""
        if not isinstance(value, str):
            return value

        normalized = value.strip().lower()
        aliases = {
            "answer": AgentActionType.SEND_MESSAGE,
            "message": AgentActionType.SEND_MESSAGE,
            "reply": AgentActionType.SEND_MESSAGE,
            "respond": AgentActionType.SEND_MESSAGE,
            "send": AgentActionType.SEND_MESSAGE,
            "chat": AgentActionType.SEND_MESSAGE,
            "none": AgentActionType.IGNORE,
            "no_reply": AgentActionType.IGNORE,
            "skip": AgentActionType.IGNORE,
            "photo": AgentActionType.TAKE_PHOTO,
        }
        return aliases.get(normalized, normalized)

    @field_validator("emotion", mode="before")
    @classmethod
    def normalize_emotion(cls, value: object) -> AgentEmotion | str:
        """Map common model-invented emotion labels to the supported set."""
        if not isinstance(value, str):
            return value

        normalized = value.strip().lower()
        aliases = {
            "helpful": AgentEmotion.CARING,
            "friendly": AgentEmotion.HAPPY,
            "kind": AgentEmotion.CARING,
            "supportive": AgentEmotion.CARING,
            "calm": AgentEmotion.NEUTRAL,
            "curious": AgentEmotion.PLAYFUL,
        }
        return aliases.get(normalized, normalized)

    @field_validator("messages", mode="before")
    @classmethod
    def clean_messages(cls, value: object) -> list[str]:
        """Drop empty messages and cap each message to Telegram-friendly length."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("messages must be a list")

        messages: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            if _looks_like_service_leak(text):
                text = SERVICE_LEAK_FALLBACK_MESSAGE
            messages.append(text[:MAX_AGENT_MESSAGE_LENGTH])
        return messages

    @field_validator("tool_input", mode="before")
    @classmethod
    def normalize_tool_input(cls, value: object) -> dict:
        """Accept empty tool input variants returned by smaller local models."""
        if value is None or value == "":
            return {}
        if not isinstance(value, dict):
            raise ValueError("tool_input must be an object")
        return value

    @field_validator("delay_seconds", mode="before")
    @classmethod
    def normalize_delay_seconds(cls, value: object) -> int | object:
        """Coerce simple numeric delay variants to bounded whole seconds."""
        if value is None or value == "":
            return 0
        if isinstance(value, str):
            value = value.strip()
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value
        return max(0, min(60, round(numeric_value)))

    @model_validator(mode="after")
    def validate_action_payload(self) -> "AgentDecision":
        """Ensure send_message decisions have something to send."""
        if self.action == AgentActionType.SEND_MESSAGE and not self.messages:
            raise ValueError("send_message action requires at least one message")
        if self.action == AgentActionType.IGNORE:
            self.messages = []
        return self

    def normalized_messages(self) -> list[str]:
        """Return cleaned messages ready for Telegram."""
        return [message.strip()[:MAX_AGENT_MESSAGE_LENGTH] for message in self.messages if message.strip()]


def _looks_like_service_leak(text: str) -> bool:
    """Return whether a user-visible message leaked internal JSON/tool instructions."""
    return any(pattern.search(text) for pattern in SERVICE_LEAK_PATTERNS)
