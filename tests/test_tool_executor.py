from dataclasses import dataclass

from app.agent.schemas import AgentDecision
from app.agent.tool_executor import ToolExecutor


@dataclass
class FakeTelegramMessage:
    message_id: int


class FakeBot:
    def __init__(self) -> None:
        self.sent_actions: list[tuple[int, str]] = []
        self.sent_messages: list[tuple[int, str]] = []

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.sent_actions.append((chat_id, action))

    async def send_message(self, chat_id: int, text: str) -> FakeTelegramMessage:
        self.sent_messages.append((chat_id, text))
        return FakeTelegramMessage(message_id=len(self.sent_messages))


async def test_send_message_sends_all_messages() -> None:
    bot = FakeBot()
    executor = ToolExecutor(bot=bot)
    decision = AgentDecision(action="send_message", messages=["one", "two"])

    result = await executor.execute(decision, telegram_chat_id=123)

    assert result.status == "success"
    assert bot.sent_messages == [(123, "one"), (123, "two")]
    assert result.output["sent_messages"] == [
        {"message_id": 1, "text": "one"},
        {"message_id": 2, "text": "two"},
    ]


async def test_ignore_sends_nothing() -> None:
    bot = FakeBot()
    executor = ToolExecutor(bot=bot)
    decision = AgentDecision(action="ignore")

    result = await executor.execute(decision, telegram_chat_id=123)

    assert result.output == {"ignored": True}
    assert bot.sent_messages == []


async def test_stub_tool_returns_stub_result() -> None:
    bot = FakeBot()
    executor = ToolExecutor(bot=bot)
    decision = AgentDecision(action="remember", tool_input={"value": "x"})

    result = await executor.execute(decision, telegram_chat_id=123)

    assert result.status == "stub"
    assert result.output["stub"] is True
