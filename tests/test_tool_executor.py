from dataclasses import dataclass

from app.agent.schemas import AgentDecision
from app.agent.tool_executor import ToolExecutor
from app.image_generation.stable_waifu_provider import ImageGenerationResult
from app.memory.diary import DiaryServiceResult


@dataclass
class FakeTelegramMessage:
    message_id: int


class FakeBot:
    def __init__(self) -> None:
        self.sent_actions: list[tuple[int, str]] = []
        self.sent_messages: list[tuple[int, str]] = []
        self.sent_photos: list[dict] = []

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.sent_actions.append((chat_id, action))

    async def send_message(self, chat_id: int, text: str) -> FakeTelegramMessage:
        self.sent_messages.append((chat_id, text))
        return FakeTelegramMessage(message_id=len(self.sent_messages))

    async def send_photo(self, chat_id: int, photo, caption: str | None = None) -> FakeTelegramMessage:
        self.sent_photos.append({"chat_id": chat_id, "photo": photo, "caption": caption})
        return FakeTelegramMessage(message_id=len(self.sent_photos))


class FakeDiaryService:
    def __init__(self) -> None:
        self.remembered: list[dict] = []

    async def create_daily_summary(self, session, user_id: int) -> DiaryServiceResult:
        return DiaryServiceResult(status="created", created_count=1, day_summary="Summary")

    async def remember_manual(self, session, user_id: int, content: str, **kwargs) -> str:
        self.remembered.append({"user_id": user_id, "content": content, **kwargs})
        return "diary_entry:1"


class FakeImageService:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.calls: list[dict[str, str]] = []
        self.base_prompt_updates: list[dict[str, str]] = []

    async def generate(
        self,
        scene: str,
        mood: str = "",
        style: str = "",
        negative: str = "",
    ) -> ImageGenerationResult:
        self.calls.append({"scene": scene, "mood": mood, "style": style, "negative": negative})
        return ImageGenerationResult(
            provider="fake",
            image_path=self.image_path,
            prompt=", ".join(part for part in [scene, mood, style] if part),
            negative_prompt=negative,
            model="FakeModel",
            preset="FakePreset",
            telegram_message_id=42,
            metadata={"source": "test"},
        )

    def update_base_prompt(
        self,
        add_tags: str = "",
        remove_tags: str = "",
        set_tags: str = "",
    ) -> dict[str, str]:
        self.base_prompt_updates.append(
            {"add_tags": add_tags, "remove_tags": remove_tags, "set_tags": set_tags}
        )
        return {"provider": "fake", "base_tags": set_tags or add_tags}


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


async def test_remember_tool_creates_manual_memory() -> None:
    bot = FakeBot()
    diary_service = FakeDiaryService()
    executor = ToolExecutor(bot=bot, diary_service=diary_service)
    decision = AgentDecision(action="remember", tool_input={"value": "x", "title": "T"})

    result = await executor.execute(decision, telegram_chat_id=123, session=object(), user_id=7)

    assert result.status == "success"
    assert result.output["memory_id"] == "diary_entry:1"
    assert diary_service.remembered[0]["content"] == "x"
    assert diary_service.remembered[0]["title"] == "T"


async def test_sleep_tool_calls_diary_service() -> None:
    bot = FakeBot()
    executor = ToolExecutor(bot=bot, diary_service=FakeDiaryService())
    decision = AgentDecision(action="sleep")

    result = await executor.execute(decision, telegram_chat_id=123, session=object(), user_id=1)

    assert result.status == "created"
    assert result.output["created_count"] == 1


async def test_take_photo_generates_and_sends_photo(tmp_path) -> None:
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"png")
    bot = FakeBot()
    image_service = FakeImageService(str(image_path))
    executor = ToolExecutor(
        bot=bot,
        image_service=image_service,
    )
    decision = AgentDecision(
        action="take_photo",
        messages=["caption from model"],
        tool_input={"description": "smiling", "mood": "happy", "style": "anime selfie"},
    )

    result = await executor.execute(decision, telegram_chat_id=123)

    assert result.status == "success"
    assert bot.sent_photos[0]["chat_id"] == 123
    assert bot.sent_photos[0]["caption"] == "caption from model"
    assert image_service.calls[0] == {
        "scene": "smiling",
        "mood": "happy",
        "style": "anime selfie",
        "negative": "",
    }
    assert result.output["image_path"] == str(image_path)
    assert result.output["caption"] == "caption from model"
    assert result.output["provider"] == "fake"
    assert result.output["prompt"] == "smiling, happy, anime selfie"
    assert result.output["model"] == "FakeModel"


async def test_take_photo_prefers_scene_tags(tmp_path) -> None:
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"png")
    image_service = FakeImageService(str(image_path))
    executor = ToolExecutor(bot=FakeBot(), image_service=image_service)
    decision = AgentDecision(
        action="take_photo",
        messages=["caption"],
        tool_input={"scene_tags": "selfie, cozy room", "description": "ignored"},
    )

    result = await executor.execute(decision, telegram_chat_id=123)

    assert result.status == "success"
    assert image_service.calls[0]["scene"] == "selfie, cozy room"


async def test_take_photo_accepts_structured_tag_categories(tmp_path) -> None:
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"png")
    image_service = FakeImageService(str(image_path))
    executor = ToolExecutor(bot=FakeBot(), image_service=image_service)
    decision = AgentDecision(
        action="take_photo",
        messages=["caption"],
        tool_input={
            "scene_tags": "selfie",
            "emotion_tags": ["sleepy"],
            "environment_tags": ["cozy room"],
            "lighting_tags": ["soft lighting"],
            "camera_tags": ["phone camera"],
        },
    )

    result = await executor.execute(decision, telegram_chat_id=123)

    assert result.status == "success"
    assert image_service.calls[0] == {
        "scene": "selfie",
        "mood": "sleepy, cozy room, soft lighting",
        "style": "phone camera",
        "negative": "",
    }


async def test_take_photo_sends_pre_photo_messages_and_caption(tmp_path) -> None:
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"png")
    bot = FakeBot()
    image_service = FakeImageService(str(image_path))
    executor = ToolExecutor(bot=bot, image_service=image_service)
    decision = AgentDecision(
        action="take_photo",
        messages=["Секунду...", "Сейчас покажу."],
        tool_input={"scene_tags": "selfie, cozy room", "caption": "для тебя"},
    )

    result = await executor.execute(decision, telegram_chat_id=123)

    assert result.status == "success"
    assert bot.sent_messages == [(123, "Секунду..."), (123, "Сейчас покажу.")]
    assert bot.sent_photos[0]["caption"] == "для тебя"
    assert result.output["sent_messages"] == [
        {"message_id": 1, "text": "Секунду..."},
        {"message_id": 2, "text": "Сейчас покажу."},
    ]


async def test_update_image_base_prompt_updates_runtime_prompt(tmp_path) -> None:
    image_service = FakeImageService(str(tmp_path / "photo.png"))
    bot = FakeBot()
    executor = ToolExecutor(bot=bot, image_service=image_service)
    decision = AgentDecision(
        action="update_image_base_prompt",
        messages=["Готово, я обновила базовый образ."],
        tool_input={"add_tags": "black choker", "remove_tags": "blue nails"},
    )

    result = await executor.execute(decision, telegram_chat_id=123)

    assert result.status == "success"
    assert image_service.base_prompt_updates == [
        {"add_tags": "black choker", "remove_tags": "blue nails", "set_tags": ""}
    ]
    assert bot.sent_messages == [(123, "Готово, я обновила базовый образ.")]
