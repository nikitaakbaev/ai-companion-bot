from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

from app.image_generation.generation_queue import ImageGenerationQueue
from app.image_generation.stable_waifu_provider import (
    StableWaifuTelegramProvider,
    detect_current_aspect_ratio,
    find_ratio_button_position,
)


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
        aspect_ratio_keywords="Ratio,Соотношение",
        editor_wait_keywords="Model",
    )


def test_detect_ratio_from_text_and_button() -> None:
    message = FakeMessage("Current ratio: 16:9", [[FakeButton("9:16")]])

    assert detect_current_aspect_ratio(message) == "16:9"
    assert find_ratio_button_position(message, "9:16") == (0, 0)


async def test_set_ratio_clicks_button_and_uses_updated_message() -> None:
    updated = FakeMessage("Model current ratio: 16:9", [[FakeButton("✓ 16:9")]])
    initial = FakeMessage(
        "Current ratio: 1:1",
        [[FakeButton("16:9", lambda: userbot.messages.__setitem__(0, updated))]],
    )
    userbot = FakeUserbot(initial)
    provider = make_provider(userbot)

    selected = await provider.set_aspect_ratio(initial, "16:9")

    assert selected is True
    assert initial.clicked == [(0, 0)]
    assert detect_current_aspect_ratio(userbot.messages[0]) == "16:9"


async def test_set_ratio_opens_dynamic_ratio_menu_when_value_is_not_visible() -> None:
    selector = FakeMessage("Model ratio menu", [[FakeButton("16:9")]])
    main_editor = FakeMessage(
        "Model",
        [[FakeButton("📐 Соотношение - 1:1 >", lambda: userbot.messages.__setitem__(0, selector))]],
    )
    userbot = FakeUserbot(main_editor)
    provider = make_provider(userbot)

    selected = await provider.set_aspect_ratio(main_editor, "16:9")

    assert selected is True
    assert main_editor.clicked == [(0, 0)]
    assert selector.clicked == [(0, 0)]


async def test_ratio_can_update_after_orientation_change() -> None:
    portrait_ratios = FakeMessage("Model", [[FakeButton("9:16")]])
    landscape_ratios = FakeMessage("Model", [[FakeButton("16:9"), FakeButton("4:3")]])
    userbot = FakeUserbot(portrait_ratios)
    userbot.messages[0] = landscape_ratios
    provider = make_provider(userbot)

    selected = await provider.set_aspect_ratio(landscape_ratios, "16:9")

    assert selected is True
    assert landscape_ratios.clicked == [(0, 0)]
