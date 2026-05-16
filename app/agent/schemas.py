"""Agent schemas."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_AGENT_MESSAGE_LENGTH = 2000


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
        return [message.strip()[:MAX_AGENT_MESSAGE_LENGTH] for message in self.messages if message.strip()]
