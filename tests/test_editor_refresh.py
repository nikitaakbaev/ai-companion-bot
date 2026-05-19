from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

from app.image_generation.generation_queue import ImageGenerationQueue
from app.image_generation.stable_waifu_provider import StableWaifuTelegramProvider


@dataclass
class FakeButton:
    text: str
    on_click: Callable[[], None] | None = None


class FakeMessage:
    def __init__(self, raw_text: str, buttons: list[list[FakeButton]]) -> None:
        self.raw_text = raw_text
        self.buttons = buttons
        self.date = datetime.now(UTC)
        self.clicked: list[tuple[int, int]] = []

    async def click(self, row: int, col: int) -> None:
        self.clicked.append((row, col))
        callback = self.buttons[row][col].on_click
        if callback is not None:
            callback()


class FakeUserbot:
    def __init__(self, message: FakeMessage) -> None:
        self.messages = [message]
        self.read_count = 0

    async def get_recent_messages(self, entity: str, limit: int = 10) -> list[FakeMessage]:
        self.read_count += 1
        return self.messages[:limit]


def make_provider(userbot: FakeUserbot) -> StableWaifuTelegramProvider:
    return StableWaifuTelegramProvider(
        userbot=userbot,
        queue=ImageGenerationQueue(),
        bot_username="StableWaifuBot",
        output_dir="./out",
        timeout_seconds=1,
        poll_interval_seconds=0,
        model_name="NekoXL",
        preset_name="Modern",
        editor_wait_keywords="Model",
        landscape_keywords="Landscape",
        portrait_keywords="Portrait",
    )


async def test_provider_reads_updated_editor_after_model_click() -> None:
    updated = FakeMessage("Model updated", [[FakeButton("Modern")]])
    initial = FakeMessage(
        "Model 1 / 1",
        [[FakeButton("NekoXL", lambda: userbot.messages.__setitem__(0, updated))]],
    )
    userbot = FakeUserbot(initial)
    provider = make_provider(userbot)

    selected = await provider.select_model("NekoXL")

    assert selected is True
    assert userbot.messages[0] is updated
    assert userbot.read_count >= 2


async def test_provider_reads_updated_editor_after_orientation_click() -> None:
    updated = FakeMessage("Model updated", [[FakeButton("Landscape")]])
    initial = FakeMessage(
        "Model",
        [[FakeButton("Portrait", lambda: userbot.messages.__setitem__(0, updated))]],
    )
    userbot = FakeUserbot(initial)
    provider = make_provider(userbot)

    selected = await provider.set_orientation(initial, "landscape")

    assert selected is True
    assert userbot.messages[0] is updated
    assert userbot.read_count >= 1
