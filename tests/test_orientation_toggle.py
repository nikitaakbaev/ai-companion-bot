from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

from app.image_generation.generation_queue import ImageGenerationQueue
from app.image_generation.stable_waifu_provider import (
    StableWaifuTelegramProvider,
    detect_current_orientation,
)


@dataclass
class FakeButton:
    text: str
    on_click: Callable[[], None] | None = None


class FakeMessage:
    def __init__(self, buttons: list[list[FakeButton]]) -> None:
        self.raw_text = "Model"
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

    async def get_recent_messages(self, entity: str, limit: int = 10) -> list[FakeMessage]:
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
        landscape_keywords="Landscape",
        portrait_keywords="Portrait",
        editor_wait_keywords="Model",
    )


async def test_portrait_switches_to_landscape() -> None:
    landscape = FakeMessage([[FakeButton("Landscape")]])
    portrait = FakeMessage([[FakeButton("Portrait", lambda: userbot.messages.__setitem__(0, landscape))]])
    userbot = FakeUserbot(portrait)
    provider = make_provider(userbot)

    switched = await provider.set_orientation(portrait, "landscape")

    assert switched is True
    assert portrait.clicked == [(0, 0)]
    assert detect_current_orientation(
        userbot.messages[0],
        landscape_keywords=["Landscape"],
        portrait_keywords=["Portrait"],
    ) == "landscape"


async def test_landscape_switches_to_portrait() -> None:
    portrait = FakeMessage([[FakeButton("Portrait")]])
    landscape = FakeMessage([[FakeButton("Landscape", lambda: userbot.messages.__setitem__(0, portrait))]])
    userbot = FakeUserbot(landscape)
    provider = make_provider(userbot)

    switched = await provider.set_orientation(landscape, "portrait")

    assert switched is True
    assert landscape.clicked == [(0, 0)]


async def test_orientation_noop_when_already_matching() -> None:
    landscape = FakeMessage([[FakeButton("Landscape")]])
    userbot = FakeUserbot(landscape)
    provider = make_provider(userbot)

    switched = await provider.set_orientation(landscape, "landscape")

    assert switched is True
    assert landscape.clicked == []
