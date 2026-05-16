"""Generated image selector placeholders."""


def select_best_image(paths: list[str]) -> str | None:
    """Select the first generated image for now."""
    return paths[0] if paths else None

