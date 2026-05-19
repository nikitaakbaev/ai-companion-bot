"""Tool definitions for the JSON agent loop."""

AVAILABLE_TOOLS = [
    {
        "name": "send_message",
        "description": "Send one or more Telegram messages to the user.",
        "implemented": True,
    },
    {
        "name": "ignore",
        "description": "Do not reply to the user, but save the agent action.",
        "implemented": True,
    },
    {
        "name": "remember",
        "description": "Save an important manual memory to the diary and vector memory.",
        "implemented": True,
    },
    {
        "name": "read_diary",
        "description": "Read recent diary entries for tool output.",
        "implemented": True,
    },
    {
        "name": "sleep",
        "description": "Create diary reflection entries from recent conversation history.",
        "implemented": True,
    },
    {
        "name": "take_photo",
        "description": (
            "Generate a Stable Waifu image/selfie. Use tool_input.scene_tags with short "
            "lowercase comma-separated anime tags only."
        ),
        "implemented": True,
    },
    {
        "name": "update_image_base_prompt",
        "description": (
            "Update the runtime base Stable Waifu character tags when the user explicitly "
            "asks for a permanent visual identity/base prompt change."
        ),
        "implemented": True,
    },
    {
        "name": "analyze_image",
        "description": "Analyze an incoming image. Requires vision configuration.",
        "implemented": False,
    },
]


def available_tools() -> list[dict]:
    """Return tool schemas for the LLM."""
    return AVAILABLE_TOOLS
