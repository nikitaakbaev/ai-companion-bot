"""Tool definitions for the JSON agent loop."""

AVAILABLE_TOOLS = [
    {
        "name": "send_message",
        "description": "Отправить одно или несколько сообщений пользователю в Telegram.",
        "implemented": True,
    },
    {
        "name": "ignore",
        "description": "Не отвечать пользователю.",
        "implemented": True,
    },
    {
        "name": "remember",
        "description": "Сохранить важное воспоминание. Пока заглушка.",
        "implemented": False,
    },
    {
        "name": "read_diary",
        "description": "Прочитать дневник. Пока заглушка.",
        "implemented": False,
    },
    {
        "name": "sleep",
        "description": "Создать дневниковое сжатие. Пока заглушка.",
        "implemented": False,
    },
    {
        "name": "take_photo",
        "description": "Сгенерировать изображение/селфи. Пока заглушка.",
        "implemented": False,
    },
    {
        "name": "analyze_image",
        "description": "Проанализировать изображение. Пока заглушка.",
        "implemented": False,
    },
]


def available_tools() -> list[dict]:
    """Return tool schemas for the LLM in later stages."""
    return AVAILABLE_TOOLS
