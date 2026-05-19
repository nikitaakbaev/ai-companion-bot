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
    def __init__(self) -> None:
        self.messages: list[FakeMessage] = []
        self.read_count = 0

    async def get_recent_messages(self, entity: str, limit: int = 10) -> list[FakeMessage]:
        self.read_count += 1
        return self.messages[:limit]

    async def send_message(self, entity: str, text: str):
        return None

    async def download_media(self, message, output_dir: str) -> str:
        return f"{output_dir}/image.png"


def make_provider(userbot: FakeUserbot, max_switches: int = 10) -> StableWaifuTelegramProvider:
    return StableWaifuTelegramProvider(
        userbot=userbot,
        queue=ImageGenerationQueue(),
        bot_username="StableWaifuBot",
        output_dir="./out",
        timeout_seconds=1,
        poll_interval_seconds=0,
        model_name="SkyVXL",
        preset_name="Opus",
        model_search_max_page_switches=max_switches,
        model_menu_keywords="Model",
        next_page_keywords="Next",
        editor_wait_keywords="Model",
    )


async def test_model_search_finds_model_on_page_2() -> None:
    userbot = FakeUserbot()
    page_2 = FakeMessage("Model 2 / 3", [[FakeButton("SkyVXL")]])
    page_1 = FakeMessage("Model 1 / 3", [[FakeButton("NekoXL")], [FakeButton("Next", lambda: userbot.messages.__setitem__(0, page_2))]])
    userbot.messages = [page_1]

    found = await make_provider(userbot).select_model("SkyVXL")

    assert found is True
    assert page_1.clicked == [(1, 0)]
    assert page_2.clicked == [(0, 0)]


async def test_model_menu_opens_before_searching_pages() -> None:
    userbot = FakeUserbot()
    page_2 = FakeMessage("Model 2 / 2", [[FakeButton("SkyVXL")]])
    page_1 = FakeMessage(
        "Model 1 / 2",
        [[FakeButton("NekoXL")], [FakeButton("Next", lambda: userbot.messages.__setitem__(0, page_2))]],
    )
    main_editor = FakeMessage(
        "New art",
        [[FakeButton("⚙️ Model: NekoXL", lambda: userbot.messages.__setitem__(0, page_1))]],
    )
    userbot.messages = [main_editor]

    found = await make_provider(userbot).select_model("SkyVXL")

    assert found is True
    assert main_editor.clicked == [(0, 0)]
    assert page_1.clicked == [(1, 0)]
    assert page_2.clicked == [(0, 0)]


async def test_model_search_finds_model_on_page_3() -> None:
    userbot = FakeUserbot()
    page_3 = FakeMessage("Model 3 / 3", [[FakeButton("SkyVXL")]])
    page_2 = FakeMessage("Model 2 / 3", [[FakeButton("BlueXL")], [FakeButton("Next", lambda: userbot.messages.__setitem__(0, page_3))]])
    page_1 = FakeMessage("Model 1 / 3", [[FakeButton("NekoXL")], [FakeButton("Next", lambda: userbot.messages.__setitem__(0, page_2))]])
    userbot.messages = [page_1]

    found = await make_provider(userbot).select_model("SkyVXL")

    assert found is True
    assert page_1.clicked == [(1, 0)]
    assert page_2.clicked == [(1, 0)]
    assert page_3.clicked == [(0, 0)]


async def test_model_search_stops_on_repeated_page() -> None:
    userbot = FakeUserbot()
    page_1 = FakeMessage("Model 1 / 3", [[FakeButton("NekoXL")], [FakeButton("Next")]])
    userbot.messages = [page_1]

    found = await make_provider(userbot, max_switches=10).select_model("SkyVXL")

    assert found is False
    assert page_1.clicked == [(1, 0)]
    assert userbot.read_count < 10
