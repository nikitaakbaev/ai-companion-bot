"""Stable Waifu Telegram provider powered by a Telethon userbot."""

import asyncio
import difflib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.image_generation.generation_queue import ImageGenerationQueue
from app.image_generation.telegram_userbot import TelegramUserbotClient

logger = logging.getLogger(__name__)

STABLE_WAIFU_MODELS = ("NekoXL", "KanaXL", "BlueXL", "VersaXL", "EpicXL", "SkyVXL")
STABLE_WAIFU_PRESETS = ("Modern", "Opus", "Eclair")
PAGE_RE = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")
RATIO_RE = re.compile(r"(?<!\d)(\d+:\d+)(?!\d)")
SELECTED_MARKERS = ("✓", "✅", "●", "🔘", "[x]", "(x)")
IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
STABLE_WAIFU_RESULT_TEXT_MARKERS = (
    "генерация завершена",
    "generation completed",
    "результата",
    "result",
)
STABLE_WAIFU_STATUS_TEXT_MARKERS = (
    "генерация начинается",
    "это займёт",
    "это займет",
    "generation starts",
    "generation started",
    "cancel",
    "отменить",
)


class StableWaifuGenerationError(Exception):
    """Raised when Stable Waifu generation fails."""


@dataclass(frozen=True)
class ImageGenerationResult:
    """Result returned by an image generation provider."""

    provider: str
    image_path: str
    prompt: str
    negative_prompt: str | None = None
    model: str | None = None
    preset: str | None = None
    telegram_message_id: int | None = None
    metadata: dict | None = None


class StableWaifuTelegramProvider:
    """Generates images by driving Stable Waifu through Telegram inline buttons."""

    def __init__(
        self,
        userbot: TelegramUserbotClient,
        queue: ImageGenerationQueue,
        bot_username: str,
        output_dir: str,
        timeout_seconds: int,
        poll_interval_seconds: int,
        model_name: str,
        preset_name: str,
        orientation: str = "landscape",
        aspect_ratio: str = "16:9",
        model_search_max_page_switches: int = 10,
        landscape_keywords: str | list[str] = "",
        portrait_keywords: str | list[str] = "",
        model_menu_keywords: str | list[str] = "",
        preset_menu_keywords: str | list[str] = "",
        aspect_ratio_keywords: str | list[str] = "",
        next_page_keywords: str | list[str] = "",
        previous_page_keywords: str | list[str] = "",
        start_keywords: str | list[str] = "",
        save_keywords: str | list[str] = "",
        editor_wait_keywords: str | list[str] = "",
        default_landscape_ratio: str = "16:9",
        default_portrait_ratio: str = "9:16",
        click_start_button: bool = True,
        click_save_button: bool = True,
    ) -> None:
        self.userbot = userbot
        self.queue = queue
        self.bot_username = _normalize_bot_username(bot_username)
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.model_name = model_name
        self.preset_name = preset_name
        self.orientation = orientation.strip().lower()
        self.aspect_ratio = aspect_ratio.strip()
        self.model_search_max_page_switches = model_search_max_page_switches
        self.landscape_keywords = parse_keywords(landscape_keywords)
        self.portrait_keywords = parse_keywords(portrait_keywords)
        self.model_menu_keywords = parse_keywords(model_menu_keywords)
        self.preset_menu_keywords = parse_keywords(preset_menu_keywords)
        self.aspect_ratio_keywords = parse_keywords(aspect_ratio_keywords)
        self.next_page_keywords = parse_keywords(next_page_keywords)
        self.previous_page_keywords = parse_keywords(previous_page_keywords)
        self.start_keywords = parse_keywords(start_keywords)
        self.save_keywords = parse_keywords(save_keywords)
        self.editor_wait_keywords = parse_keywords(editor_wait_keywords)
        self.default_landscape_ratio = default_landscape_ratio
        self.default_portrait_ratio = default_portrait_ratio
        self.click_start_button = click_start_button
        self.click_save_button = click_save_button
        self._current_start_time: datetime | None = None

    async def generate(self, prompt: str, negative_prompt: str | None = None) -> ImageGenerationResult:
        """Generate an image through Stable Waifu."""
        try:
            return await self.queue.run_exclusive(self._generate(prompt, negative_prompt))
        except Exception:
            logger.exception(
                "Stable Waifu generation failed",
                extra={
                    "bot_username": self.bot_username,
                    "model": self.model_name,
                    "preset": self.preset_name,
                    "orientation": self.orientation,
                    "aspect_ratio": self.aspect_ratio,
                    "timeout_seconds": self.timeout_seconds,
                },
            )
            raise

    async def _generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> ImageGenerationResult:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise StableWaifuGenerationError("Stable Waifu prompt must not be empty")

        start_time = datetime.now(UTC)
        self._current_start_time = start_time
        logger.info(
            "Stable Waifu generation started",
            extra={
                "bot_username": self.bot_username,
                "model": self.model_name,
                "preset": self.preset_name,
                "orientation": self.orientation,
                "aspect_ratio": self.aspect_ratio,
                "prompt_length": len(clean_prompt),
                "negative_prompt_length": len(negative_prompt or ""),
            },
        )
        logger.info("Sending prompt to Stable Waifu bot")
        await self.userbot.send_message(self.bot_username, clean_prompt)
        logger.info("Waiting for Stable Waifu editor message")
        await self.wait_for_editor_message(start_time)

        logger.info("Selecting Stable Waifu model")
        if not await self.select_model(self.model_name):
            logger.warning(
                "Stable Waifu model not found; using current model",
                extra={"requested_model": self.model_name},
            )
        logger.info("Selecting Stable Waifu preset")
        if not await self.select_preset(self.preset_name):
            logger.warning(
                "Stable Waifu preset not found; using current preset",
                extra={"requested_preset": self.preset_name},
            )

        editor = await self.get_latest_editor_message()
        logger.info("Setting Stable Waifu orientation")
        if not await self.set_orientation(editor, self.orientation):
            logger.warning(
                "Stable Waifu orientation was not switched",
                extra={"desired_orientation": self.orientation},
            )

        editor = await self.get_latest_editor_message()
        desired_ratio = self.aspect_ratio or self._default_ratio_for_orientation()
        logger.info("Setting Stable Waifu aspect ratio")
        if not await self.set_aspect_ratio(editor, desired_ratio):
            logger.warning(
                "Stable Waifu aspect ratio was not found",
                extra={"desired_ratio": desired_ratio},
            )

        editor = await self.get_latest_editor_message()
        if self.click_save_button:
            logger.info("Looking for Stable Waifu Save button")
            if await click_button_by_keywords(editor, self.save_keywords, label="Save"):
                logger.info("Save clicked")
                await self.get_latest_editor_message()
            else:
                logger.warning("Stable Waifu Save button not found; continuing")

        if self.click_start_button:
            editor = await self.get_latest_editor_message()
            logger.info("Looking for Stable Waifu Start button")
            await self.click_required(editor, self.start_keywords, "Start")
            logger.info("Start clicked")

        logger.info("Waiting image...")
        image_message = await self.wait_for_image_message(start_time)
        image_path = await self.userbot.download_media(image_message, self.output_dir)
        if not image_path:
            raise StableWaifuGenerationError("Stable Waifu returned an image but download failed")
        logger.info("Image downloaded", extra={"image_path": str(image_path)})

        return ImageGenerationResult(
            provider="stable_waifu_telegram",
            image_path=str(image_path),
            prompt=clean_prompt,
            negative_prompt=negative_prompt,
            model=self.model_name,
            preset=self.preset_name,
            telegram_message_id=getattr(image_message, "id", None),
            metadata={
                "bot_username": self.bot_username,
                "output_dir": str(Path(self.output_dir)),
                "orientation": self.orientation,
                "aspect_ratio": desired_ratio,
            },
        )

    async def select_model(self, model_name: str) -> bool:
        """Find and click a model button, moving through editor pages if needed."""
        if model_name not in STABLE_WAIFU_MODELS:
            raise StableWaifuGenerationError(_unsupported_value_message(
                label="Stable Waifu model",
                value=model_name,
                supported=STABLE_WAIFU_MODELS,
            ))

        logger.info("Searching model: %s", model_name)
        editor = await self.get_latest_editor_message()
        if button_with_keywords_contains_value(editor, self.model_menu_keywords, model_name):
            logger.info("Model already selected: %s", model_name)
            return True
        if self.model_menu_keywords and await click_button_by_keywords(
            editor,
            self.model_menu_keywords,
            label="model menu",
        ):
            logger.info("Opened Stable Waifu model selector")
            await self.get_latest_editor_message()

        visited_pages: set[int] = set()
        switch_count = 0
        while switch_count <= self.model_search_max_page_switches:
            editor_message = await self.get_latest_editor_message()
            page_info = extract_page_info(editor_message)
            if page_info is not None:
                current_page, total_pages = page_info
                logger.info("Current page: %s/%s", current_page, total_pages)
                if current_page in visited_pages:
                    logger.warning("Stable Waifu model search stopped on repeated page")
                    break
                visited_pages.add(current_page)

            if await click_button_by_keywords(editor_message, [model_name], label=f"model {model_name}"):
                logger.info("Model selected: %s", model_name)
                await self.get_latest_editor_message()
                return True

            logger.info("Model not found")
            if switch_count >= self.model_search_max_page_switches:
                break
            if not await click_button_by_keywords(editor_message, self.next_page_keywords, label="next page"):
                logger.warning(
                    "Stable Waifu next page button not found during model search",
                    extra={
                        "requested_model": model_name,
                        "page_switches": switch_count,
                    },
                )
                break

            logger.info("Switching to next page")
            switch_count += 1
            await self._sleep_after_click()

        return False

    async def select_preset(self, preset_name: str) -> bool:
        """Click the requested preset button."""
        if preset_name not in STABLE_WAIFU_PRESETS:
            raise StableWaifuGenerationError(_unsupported_value_message(
                label="Stable Waifu preset",
                value=preset_name,
                supported=STABLE_WAIFU_PRESETS,
            ))
        editor = await self.get_latest_editor_message()
        if button_with_keywords_contains_value(editor, self.preset_menu_keywords, preset_name):
            logger.info("Preset already selected: %s", preset_name)
            return True
        if self.preset_menu_keywords and await click_button_by_keywords(
            editor,
            self.preset_menu_keywords,
            label="preset menu",
        ):
            logger.info("Opened Stable Waifu preset selector")
            editor = await self.get_latest_editor_message()
        if await click_button_by_keywords(editor, [preset_name], label=f"preset {preset_name}"):
            logger.info("Preset selected: %s", preset_name)
            await self.get_latest_editor_message()
            return True
        return False

    async def set_orientation(
        self,
        editor_message: Any,
        desired_orientation: str,
    ) -> bool:
        """Switch Landscape/Portrait only when the current orientation differs."""
        desired = desired_orientation.strip().lower()
        if desired not in {"landscape", "portrait"}:
            logger.warning("Unsupported Stable Waifu orientation: %s", desired_orientation)
            return False

        current = detect_current_orientation(
            editor_message,
            landscape_keywords=self.landscape_keywords,
            portrait_keywords=self.portrait_keywords,
        )
        logger.info("Current orientation: %s", current)
        logger.info("Desired orientation: %s", desired)
        if current == desired:
            return True

        logger.info("Switching orientation")
        current_keywords = (
            self.landscape_keywords
            if current == "landscape"
            else self.portrait_keywords
            if current == "portrait"
            else []
        )
        desired_keywords = self.landscape_keywords if desired == "landscape" else self.portrait_keywords
        if not await click_button_by_keywords(
            editor_message,
            current_keywords or desired_keywords,
            label="orientation toggle",
        ):
            logger.warning(
                "Stable Waifu orientation toggle button not found",
                extra={
                    "current_orientation": current,
                    "desired_orientation": desired,
                    "keywords": current_keywords or desired_keywords,
                },
            )
            return False

        updated = await self.get_latest_editor_message()
        updated_orientation = detect_current_orientation(
            updated,
            landscape_keywords=self.landscape_keywords,
            portrait_keywords=self.portrait_keywords,
        )
        return updated_orientation == desired

    async def set_aspect_ratio(
        self,
        editor_message: Any,
        desired_ratio: str,
    ) -> bool:
        """Select an aspect ratio button when available."""
        clean_ratio = desired_ratio.strip()
        if not clean_ratio:
            return False
        if _is_ratio_selected(editor_message, clean_ratio):
            logger.info("Aspect ratio set: %s", clean_ratio)
            return True

        position = find_ratio_button_position(editor_message, clean_ratio)
        if position is None and self.aspect_ratio_keywords:
            if await click_button_by_keywords(
                editor_message,
                self.aspect_ratio_keywords,
                label="aspect ratio menu",
            ):
                logger.info("Opened Stable Waifu aspect ratio selector")
                editor_message = await self.get_latest_editor_message()
                position = find_ratio_button_position(editor_message, clean_ratio)
        if position is None:
            logger.warning(
                "Stable Waifu aspect ratio button not found",
                extra={"desired_ratio": clean_ratio},
            )
            return False
        row, col = position
        logger.info(
            "Clicking Stable Waifu aspect ratio button",
            extra={"row": row, "col": col, "desired_ratio": clean_ratio},
        )
        await editor_message.click(row, col)
        updated = await self.get_latest_editor_message()
        if _message_contains_ratio(updated, clean_ratio):
            logger.info("Aspect ratio set: %s", clean_ratio)
            return True
        return False

    async def click_required(
        self,
        message: Any,
        keywords: tuple[str, ...] | list[str],
        label: str,
    ) -> None:
        """Click a button and fail with a clear error if it is missing."""
        if not await click_button_by_keywords(message, list(keywords), label=label):
            raise StableWaifuGenerationError(
                f"Stable Waifu button not found: {label}. "
                f"Keywords: {', '.join(str(keyword) for keyword in keywords)}. "
                f"Visible buttons: {_compact_buttons(message)}"
            )

    async def wait_for_editor_message(self, start_time: datetime):
        """Wait until Stable Waifu sends or edits an editor message with inline buttons."""
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        attempt = 0
        while asyncio.get_running_loop().time() < deadline:
            attempt += 1
            messages = await self.userbot.get_recent_messages(self.bot_username, limit=15)
            for message in messages:
                if _looks_like_editor_message(message, self._editor_keywords()):
                    old_message = _is_old_message(message, start_time)
                    if old_message:
                        logger.info(
                            "Stable Waifu editor message has an old date; accepting it as edited editor",
                            extra={
                                "attempt": attempt,
                                "message_id": getattr(message, "id", None),
                                "message_date": str(getattr(message, "date", None)),
                                "buttons": _compact_buttons(message),
                            },
                        )
                    logger.info(
                        "Stable Waifu editor message found",
                        extra={
                            "attempt": attempt,
                            "message_id": getattr(message, "id", None),
                            "old_message": old_message,
                            "buttons": _compact_buttons(message),
                        },
                    )
                    return message
                if getattr(message, "buttons", None):
                    logger.info(
                        "Stable Waifu message with buttons did not match editor keywords",
                        extra={
                            "attempt": attempt,
                            "message_id": getattr(message, "id", None),
                            "old_message": _is_old_message(message, start_time),
                            "editor_keywords": self._editor_keywords(),
                            "buttons": _compact_buttons(message),
                            "text": str(getattr(message, "raw_text", "") or "")[:500],
                        },
                    )
            await asyncio.sleep(self.poll_interval_seconds)
        raise StableWaifuGenerationError(
            f"Timed out waiting for Stable Waifu editor message after {self.timeout_seconds}s. "
            f"Editor keywords: {', '.join(self._editor_keywords())}"
        )

    async def get_latest_editor_message(self):
        """Refresh the latest editor message after an inline button click."""
        await self._sleep_after_click()
        start_time = self._current_start_time or datetime.min.replace(tzinfo=UTC)
        return await self.wait_for_editor_message(start_time)

    async def wait_for_image_message(self, start_time: datetime):
        """Wait until Stable Waifu sends an image message."""
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        attempt = 0
        while asyncio.get_running_loop().time() < deadline:
            attempt += 1
            messages = await self.userbot.get_recent_messages(self.bot_username, limit=20)
            for message in messages:
                if _is_old_message(message, start_time):
                    continue
                if _is_generation_status_message(message):
                    logger.info(
                        "Stable Waifu generation status message found; waiting for result file",
                        extra={
                            "attempt": attempt,
                            "message_id": getattr(message, "id", None),
                            "text": str(getattr(message, "raw_text", "") or "")[:500],
                            "buttons": _compact_buttons(message),
                        },
                    )
                    continue
                if _has_image_media(message) or _has_result_file_reference(message):
                    logger.info(
                        "Stable Waifu image message found",
                        extra={
                            "attempt": attempt,
                            "message_id": getattr(message, "id", None),
                            "text": str(getattr(message, "raw_text", "") or "")[:500],
                            "document_name": _document_filename(message),
                        },
                    )
                    return message
                if _looks_like_editor_message(message, self._editor_keywords()):
                    logger.info(
                        "Stable Waifu editor-like message ignored while waiting for image",
                        extra={
                            "attempt": attempt,
                            "message_id": getattr(message, "id", None),
                            "text": str(getattr(message, "raw_text", "") or "")[:500],
                            "buttons": _compact_buttons(message),
                        },
                    )
                    continue
            await asyncio.sleep(self.poll_interval_seconds)
        raise StableWaifuGenerationError(
            f"Timed out waiting for Stable Waifu image after {self.timeout_seconds}s"
        )

    async def _sleep_after_click(self) -> None:
        if self.poll_interval_seconds <= 0:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(min(max(self.poll_interval_seconds, 1), 2))

    def _editor_keywords(self) -> list[str]:
        return [
            *self.editor_wait_keywords,
            *self.start_keywords,
            *self.save_keywords,
            self.model_name,
            self.preset_name,
        ]

    def _default_ratio_for_orientation(self) -> str:
        if self.orientation == "portrait":
            return self.default_portrait_ratio
        return self.default_landscape_ratio


def parse_keywords(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse comma-separated keyword settings."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def find_button_position(message: Any, keywords: list[str]) -> tuple[int, int] | None:
    """Find an inline button by partial text match and return row/column."""
    buttons = getattr(message, "buttons", None)
    if not buttons:
        return None

    normalized_keywords = [keyword.casefold() for keyword in keywords if keyword.strip()]
    for row_index, row in enumerate(buttons):
        for col_index, button in enumerate(row):
            text = str(getattr(button, "text", "") or "").casefold()
            if any(keyword in text for keyword in normalized_keywords):
                return row_index, col_index
    return None


def button_with_keywords_contains_value(
    message: Any,
    keywords: list[str],
    value: str,
) -> bool:
    """Return true when a dynamic state button contains both section keyword and current value."""
    if not keywords or not value.strip():
        return False
    buttons = getattr(message, "buttons", None) or []
    normalized_keywords = [keyword.casefold() for keyword in keywords if keyword.strip()]
    normalized_value = value.casefold()
    for row in buttons:
        for button in row:
            text = str(getattr(button, "text", "") or "").casefold()
            if normalized_value in text and any(keyword in text for keyword in normalized_keywords):
                return True
    return False


async def click_button_by_keywords(
    message: Any,
    keywords: list[str],
    label: str | None = None,
) -> bool:
    """Click an inline button selected by keyword matching."""
    position = find_button_position(message, keywords)
    if position is None:
        logger.info(
            "Stable Waifu button not found",
            extra={
                "label": label,
                "keywords": keywords,
                "visible_buttons": _compact_buttons(message),
            },
        )
        return False
    row, col = position
    logger.info(
        "Clicking Stable Waifu button",
        extra={
            "label": label,
            "row": row,
            "col": col,
            "button_text": _button_text_at(message, row, col),
            "keywords": keywords,
        },
    )
    await message.click(row, col)
    return True


def extract_page_info(message: Any) -> tuple[int, int] | None:
    """Extract pagination info like 1 / 3 from editor text or buttons."""
    combined = f"{getattr(message, 'raw_text', '') or ''}\n{_all_button_text(message)}"
    match = PAGE_RE.search(combined)
    if not match:
        return None
    current_page = int(match.group(1))
    total_pages = int(match.group(2))
    if current_page <= 0 or total_pages <= 0:
        return None
    return current_page, total_pages


def detect_current_orientation(
    message: Any,
    landscape_keywords: list[str] | None = None,
    portrait_keywords: list[str] | None = None,
) -> str | None:
    """Detect current Landscape/Portrait state from configured button keywords."""
    if landscape_keywords and find_button_position(message, landscape_keywords) is not None:
        return "landscape"
    if portrait_keywords and find_button_position(message, portrait_keywords) is not None:
        return "portrait"
    return None


def find_ratio_button_position(message: Any, desired_ratio: str) -> tuple[int, int] | None:
    """Find a button containing the requested aspect ratio."""
    buttons = getattr(message, "buttons", None) or []
    for row_index, row in enumerate(buttons):
        for col_index, button in enumerate(row):
            text = str(getattr(button, "text", "") or "")
            ratios = RATIO_RE.findall(text)
            if desired_ratio in ratios:
                return row_index, col_index
    return None


def detect_current_aspect_ratio(message: Any) -> str | None:
    """Best-effort current aspect ratio detection from editor text and selected buttons."""
    raw_text = str(getattr(message, "raw_text", "") or "")
    raw_ratios = RATIO_RE.findall(raw_text)
    if raw_ratios:
        return raw_ratios[-1]

    buttons = getattr(message, "buttons", None) or []
    button_ratios: list[str] = []
    for row in buttons:
        for button in row:
            text = str(getattr(button, "text", "") or "")
            ratios = RATIO_RE.findall(text)
            if not ratios:
                continue
            if any(marker in text for marker in SELECTED_MARKERS):
                return ratios[-1]
            button_ratios.extend(ratios)
    if len(set(button_ratios)) == 1:
        return button_ratios[0]
    return None


def _is_ratio_selected(message: Any, desired_ratio: str) -> bool:
    return detect_current_aspect_ratio(message) == desired_ratio


def _message_contains_ratio(message: Any, desired_ratio: str) -> bool:
    combined = f"{getattr(message, 'raw_text', '') or ''}\n{_all_button_text(message)}"
    return desired_ratio in RATIO_RE.findall(combined)


def _looks_like_editor_message(message: Any, editor_keywords: list[str]) -> bool:
    if not getattr(message, "buttons", None):
        return False
    text = str(getattr(message, "raw_text", "") or "").casefold()
    button_text = _all_button_text(message).casefold()
    combined = f"{text}\n{button_text}"
    normalized_keywords = [keyword.casefold() for keyword in editor_keywords if keyword.strip()]
    return any(keyword in combined for keyword in normalized_keywords)


def _all_button_text(message: Any) -> str:
    buttons = getattr(message, "buttons", None) or []
    texts: list[str] = []
    for row in buttons:
        for button in row:
            texts.append(str(getattr(button, "text", "") or ""))
    return "\n".join(texts)


def _compact_buttons(message: Any) -> str:
    buttons = getattr(message, "buttons", None) or []
    rows: list[str] = []
    for row in buttons:
        row_texts = [str(getattr(button, "text", "") or "") for button in row]
        if row_texts:
            rows.append(" | ".join(row_texts))
    return " / ".join(rows)[:1000]


def _button_text_at(message: Any, row: int, col: int) -> str | None:
    buttons = getattr(message, "buttons", None) or []
    try:
        return str(getattr(buttons[row][col], "text", "") or "")
    except IndexError:
        return None


def _unsupported_value_message(
    label: str,
    value: str,
    supported: tuple[str, ...],
) -> str:
    supported_text = ", ".join(supported)
    suggestion = difflib.get_close_matches(value, supported, n=1, cutoff=0.55)
    if suggestion:
        return (
            f"Unsupported {label}: {value}. "
            f"Did you mean {suggestion[0]}? Supported values: {supported_text}"
        )
    return f"Unsupported {label}: {value}. Supported values: {supported_text}"


def _has_image_media(message: Any) -> bool:
    if getattr(message, "photo", None) is not None:
        return True
    media = getattr(message, "media", None)
    if getattr(media, "photo", None) is not None:
        return True
    for document in _iter_documents(message):
        mime_type = str(getattr(document, "mime_type", "") or "")
        if mime_type.startswith("image/"):
            return True
        filename = _filename_from_document(document)
        if filename.lower().endswith(IMAGE_FILE_EXTENSIONS):
            return True
    return False


def _has_result_file_reference(message: Any) -> bool:
    text = str(getattr(message, "raw_text", "") or "").casefold()
    if not any(marker in text for marker in STABLE_WAIFU_RESULT_TEXT_MARKERS):
        return False
    filename = _document_filename(message).casefold()
    return filename.endswith(IMAGE_FILE_EXTENSIONS)


def _is_generation_status_message(message: Any) -> bool:
    text = str(getattr(message, "raw_text", "") or "").casefold()
    button_text = _all_button_text(message).casefold()
    combined = f"{text}\n{button_text}"
    return any(marker in combined for marker in STABLE_WAIFU_STATUS_TEXT_MARKERS)


def _document_filename(message: Any) -> str:
    for document in _iter_documents(message):
        file_name = _filename_from_document(document)
        if file_name:
            return file_name
    return ""


def _iter_documents(message: Any) -> list[Any]:
    documents: list[Any] = []
    for document in (
        getattr(message, "document", None),
        getattr(getattr(message, "media", None), "document", None),
        getattr(getattr(getattr(message, "media", None), "webpage", None), "document", None),
        getattr(getattr(message, "web_preview", None), "document", None),
    ):
        if document is not None and document not in documents:
            documents.append(document)
    return documents


def _filename_from_document(document: Any) -> str:
    file_name = str(getattr(document, "file_name", "") or "")
    if file_name:
        return file_name
    attributes = getattr(document, "attributes", None) or []
    for attribute in attributes:
        file_name = str(getattr(attribute, "file_name", "") or "")
        if file_name:
            return file_name
    return str(getattr(document, "name", "") or "")


def _is_old_message(message: Any, start_time: datetime) -> bool:
    message_date = getattr(message, "date", None)
    if not isinstance(message_date, datetime):
        return False
    if message_date.tzinfo is None:
        message_date = message_date.replace(tzinfo=UTC)
    try:
        threshold = start_time - timedelta(seconds=5)
    except OverflowError:
        threshold = start_time
    return message_date < threshold


def _normalize_bot_username(username: str) -> str:
    clean = username.strip()
    if not clean:
        raise StableWaifuGenerationError("Stable Waifu bot username must not be empty")
    return clean if clean.startswith("@") else f"@{clean}"
