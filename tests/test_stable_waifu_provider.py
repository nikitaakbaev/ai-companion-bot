import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from app.image_generation.generation_queue import ImageGenerationQueue
from app.image_generation.prompt_builder import StableWaifuPromptBuilder
from app.image_generation.stable_waifu_provider import (
    StableWaifuGenerationError,
    StableWaifuTelegramProvider,
    button_with_keywords_contains_value,
    click_button_by_keywords,
    find_button_position,
)


@dataclass
class FakeButton:
    text: str


@dataclass
class FakeDocument:
    mime_type: str
    file_name: str = ""


@dataclass
class FakeMedia:
    document: FakeDocument | None = None
    photo: object | None = None
    webpage: "FakeWebPage | None" = None


@dataclass
class FakeWebPage:
    document: FakeDocument | None = None


class FakeMessage:
    def __init__(
        self,
        buttons: list[list[FakeButton]] | None = None,
        raw_text: str = "",
        photo: object | None = None,
        document: FakeDocument | None = None,
        media: FakeMedia | None = None,
    ) -> None:
        self.buttons = buttons
        self.raw_text = raw_text
        self.photo = photo
        self.document = document
        self.media = media
        self.date = datetime.now(UTC)
        self.id = 123
        self.clicked: list[tuple[int, int]] = []

    async def click(self, row: int, col: int) -> None:
        self.clicked.append((row, col))


class FakeUserbot:
    def __init__(self, messages: list[FakeMessage] | None = None) -> None:
        self.messages = messages or []
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, entity: str, text: str) -> FakeMessage:
        self.sent.append((entity, text))
        return FakeMessage()

    async def get_recent_messages(self, entity: str, limit: int = 10) -> list[FakeMessage]:
        return self.messages[:limit]

    async def download_media(self, message, output_dir: str) -> str:
        return f"{output_dir}/image.png"


def make_provider(userbot: FakeUserbot | None = None, timeout: int = 1) -> StableWaifuTelegramProvider:
    return StableWaifuTelegramProvider(
        userbot=userbot or FakeUserbot(),
        queue=ImageGenerationQueue(),
        bot_username="StableWaifuBot",
        output_dir="./out",
        timeout_seconds=timeout,
        poll_interval_seconds=0,
        model_name="NekoXL",
        preset_name="Modern",
        model_menu_keywords="Model",
        preset_menu_keywords="Preset",
        aspect_ratio_keywords="Ratio",
        next_page_keywords="Next",
        start_keywords="Start,🎉",
        save_keywords="Save",
        editor_wait_keywords="Model,Preset",
    )


def test_find_button_position_uses_partial_case_insensitive_match() -> None:
    message = FakeMessage(
        buttons=[
            [FakeButton("KanaXL"), FakeButton("NekoXL selected")],
            [FakeButton("Modern"), FakeButton("Save")],
        ]
    )

    assert find_button_position(message, ["nekoxl"]) == (0, 1)
    assert find_button_position(message, ["missing"]) is None


def test_dynamic_state_button_can_be_detected_without_clicking() -> None:
    message = FakeMessage(buttons=[[FakeButton("⚙️ Model: NekoXL")]])

    assert button_with_keywords_contains_value(message, ["Model"], "NekoXL") is True
    assert button_with_keywords_contains_value(message, ["Model"], "SkyVXL") is False


async def test_click_button_by_keywords_clicks_found_button() -> None:
    message = FakeMessage(buttons=[[FakeButton("Opus")], [FakeButton("Start now")]])

    clicked = await click_button_by_keywords(message, ["start"])

    assert clicked is True
    assert message.clicked == [(1, 0)]


async def test_model_and_preset_selection_click_matching_buttons() -> None:
    provider = make_provider()
    message = FakeMessage(
        buttons=[[FakeButton("NekoXL")], [FakeButton("Modern")]],
        raw_text="Model 1 / 1",
    )
    provider.userbot.messages = [message]

    assert await provider.select_model("NekoXL") is True
    assert await provider.select_preset("Modern") is True

    assert message.clicked == [(0, 0), (1, 0)]


async def test_model_state_button_matching_current_model_is_noop() -> None:
    provider = make_provider()
    message = FakeMessage(
        buttons=[[FakeButton("⚙️ Model: NekoXL")]],
        raw_text="Model editor",
    )
    provider.userbot.messages = [message]

    assert await provider.select_model("NekoXL") is True
    assert message.clicked == []


async def test_unsupported_model_error_lists_supported_values() -> None:
    provider = make_provider()

    with pytest.raises(StableWaifuGenerationError) as exc_info:
        await provider.select_model("NecoXL")

    message = str(exc_info.value)
    assert "Unsupported Stable Waifu model: NecoXL" in message
    assert "Did you mean NekoXL?" in message
    assert "NekoXL, KanaXL, BlueXL, VersaXL, EpicXL, SkyVXL" in message


async def test_start_button_can_be_selected_by_keywords() -> None:
    provider = make_provider()
    message = FakeMessage(buttons=[[FakeButton("🎉 Start generation")]])

    await provider.click_required(message, ["start"], "Start")

    assert message.clicked == [(0, 0)]


async def test_wait_for_editor_message_times_out() -> None:
    provider = make_provider(userbot=FakeUserbot(messages=[]), timeout=0)

    with pytest.raises(StableWaifuGenerationError, match="editor"):
        await provider.wait_for_editor_message(datetime.now(UTC))


async def test_wait_for_editor_message_accepts_edited_old_editor_message() -> None:
    provider = make_provider()
    editor = FakeMessage(
        buttons=[[FakeButton("⚙️ Model: NekoXL")], [FakeButton("🎉 Start")]],
        raw_text="Model editor",
    )
    editor.date = datetime(2020, 1, 1, tzinfo=UTC)
    provider.userbot.messages = [editor]

    result = await provider.wait_for_editor_message(datetime.now(UTC))

    assert result is editor


async def test_wait_for_image_ignores_editor_messages() -> None:
    editor = FakeMessage(buttons=[[FakeButton("NekoXL"), FakeButton("Start")]], raw_text="Model")
    image = FakeMessage(photo=object())
    provider = make_provider(userbot=FakeUserbot(messages=[editor, image]))

    result = await provider.wait_for_image_message(datetime.now(UTC))

    assert result is image


async def test_wait_for_image_accepts_result_document_file() -> None:
    status = FakeMessage(
        buttons=[[FakeButton("💔 Отменить")]],
        raw_text="✨ Генерация начинается!\nЭто займёт ~19 секунд.",
    )
    result_file = FakeMessage(
        buttons=[[FakeButton("⚙️ Изменить"), FakeButton("🔁 Повторить")]],
        raw_text="🏁 Генерация завершена!\nНиже сообщение с файлом результата.",
        document=FakeDocument(mime_type="application/octet-stream", file_name="nekoxl-test.png"),
    )
    provider = make_provider(userbot=FakeUserbot(messages=[status, result_file]))

    result = await provider.wait_for_image_message(datetime.now(UTC))

    assert result is result_file


async def test_wait_for_image_checks_file_before_editor_like_text() -> None:
    result_file = FakeMessage(
        buttons=[[FakeButton("⚙️ Изменить"), FakeButton("🔁 Повторить")]],
        raw_text=(
            "🏁 Генерация завершена!\n"
            "#nekoxl #opus\n"
            "1girl, pale skin, selfie"
        ),
        document=FakeDocument(mime_type="application/octet-stream", file_name="nekoxl-test.png"),
    )
    provider = make_provider(userbot=FakeUserbot(messages=[result_file]))

    result = await provider.wait_for_image_message(datetime.now(UTC))

    assert result is result_file


async def test_wait_for_image_accepts_nested_webpage_document_file() -> None:
    result_file = FakeMessage(
        raw_text="🏁 Генерация завершена!\nНиже сообщение с файлом результата.",
        media=FakeMedia(
            webpage=FakeWebPage(
                document=FakeDocument(
                    mime_type="application/octet-stream",
                    file_name="nested-result.webp",
                )
            ),
        ),
    )
    provider = make_provider(userbot=FakeUserbot(messages=[result_file]))

    result = await provider.wait_for_image_message(datetime.now(UTC))

    assert result is result_file


async def test_generation_queue_runs_one_job_at_a_time() -> None:
    queue = ImageGenerationQueue()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def first() -> str:
        order.append("first-start")
        first_started.set()
        await release_first.wait()
        order.append("first-end")
        return "first"

    async def second() -> str:
        order.append("second-start")
        return "second"

    first_task = asyncio.create_task(queue.run_exclusive(first()))
    await first_started.wait()
    second_task = asyncio.create_task(queue.run_exclusive(second()))
    await asyncio.sleep(0)

    assert order == ["first-start"]

    release_first.set()
    assert await first_task == "first"
    assert await second_task == "second"
    assert order == ["first-start", "first-end", "second-start"]


def test_stable_waifu_prompt_builder_deduplicates_and_limits_tags() -> None:
    builder = StableWaifuPromptBuilder(
        base_tags="1girl, blue eyes, anime",
        negative_tags="bad hands, blurry, bad hands",
        max_length=60,
    )

    prompt = builder.build_prompt(
        scene_tags="selfie, blue eyes, cozy room",
        mood="sleepy",
        style="soft lighting",
    )

    assert prompt.positive == "masterpiece, best quality, anime, 1girl, blue eyes, selfie"
    assert prompt.negative == "bad hands, blurry"


def test_stable_waifu_prompt_builder_normalizes_scene_tags_and_drops_prose() -> None:
    builder = StableWaifuPromptBuilder(
        base_tags="1girl, blue eyes, anime",
        negative_tags="bad hands",
        max_length=300,
    )

    prompt = builder.build_prompt(
        scene_tags=(
            "Selfie, Cozy Room, A beautiful anime girl sitting near a window while softly smiling, "
            "phone camera, phone camera"
        ),
        mood="Sleepy",
        style="Soft Lighting",
    )

    assert prompt.positive == (
        "masterpiece, best quality, anime, 1girl, blue eyes, selfie, cozy room, "
        "phone camera, sleepy, soft lighting"
    )


def test_stable_waifu_prompt_builder_updates_base_tags() -> None:
    builder = StableWaifuPromptBuilder(
        base_tags="1girl, blue eyes, blue nails",
        negative_tags="bad hands",
        max_length=300,
    )

    base_tags = builder.update_base_tags(add_tags="black choker", remove_tags="blue nails")
    prompt = builder.build_prompt(scene_tags="selfie")

    assert base_tags == "1girl, blue eyes, black choker"
    assert prompt.positive == "masterpiece, best quality, anime, 1girl, blue eyes, black choker, selfie"
