"""Image prompt builder placeholders."""


def build_image_prompt(description: str, mood: str, style: str) -> str:
    """Build a simple image prompt."""
    return ", ".join(part for part in (description, mood, style) if part)

