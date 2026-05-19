"""Agent schemas."""

from enum import StrEnum
import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_AGENT_MESSAGE_LENGTH = 2000
SERVICE_LEAK_EXACT_MESSAGES = {
    "tool_input",
    "emotion",
    "delay_seconds",
    "messages",
    "thought",
}
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
    UPDATE_IMAGE_BASE_PROMPT = "update_image_base_prompt"
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
            "cozy": AgentEmotion.CARING,
            "sleepy": AgentEmotion.CARING,
            "warm": AgentEmotion.CARING,
            "affectionate": AgentEmotion.CARING,
            "shy": AgentEmotion.CARING,
            "cute": AgentEmotion.HAPPY,
            "excited": AgentEmotion.HAPPY,
            "romantic": AgentEmotion.CARING,
            "flirty": AgentEmotion.PLAYFUL,
        }
        return aliases.get(normalized, normalized)

    @field_validator("messages", mode="before")
    @classmethod
    def clean_messages(cls, value: object) -> list[str]:
        """Drop empty messages and cap each message to Telegram-friendly length."""
        if value is None:
            return []
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
                continue
            messages.append(text[:MAX_AGENT_MESSAGE_LENGTH])
        return messages

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
        messages: list[str] = []
        for message in self.messages:
            text = message.strip()[:MAX_AGENT_MESSAGE_LENGTH]
            if not text or _is_duplicate_message(text, messages):
                continue
            messages.append(text)
        return messages


def _looks_like_service_leak(text: str) -> bool:
    """Return whether a user-visible message leaked internal JSON/tool instructions."""
    normalized = text.strip().lower()
    return normalized in SERVICE_LEAK_EXACT_MESSAGES or any(
        pattern.search(text) for pattern in SERVICE_LEAK_PATTERNS
    )


def _is_duplicate_message(text: str, previous_messages: list[str]) -> bool:
    normalized = _normalize_for_similarity(text)
    if not normalized:
        return True
    for previous in previous_messages:
        previous_normalized = _normalize_for_similarity(previous)
        if normalized == previous_normalized:
            return True
        if SequenceMatcher(None, normalized, previous_normalized).ratio() >= 0.82:
            return True
    return False


def _normalize_for_similarity(text: str) -> str:
    lowered = text.casefold()
    without_punctuation = re.sub(r"[^\w\sа-яёА-ЯЁ]", " ", lowered)
    return " ".join(without_punctuation.split())
