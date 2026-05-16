"""Human-like behavior helpers."""


def calculate_typing_delay(text: str) -> float:
    """Return a simple typing delay estimate."""
    return min(max(len(text) / 30.0, 0.5), 8.0)


def should_delay_response() -> bool:
    """Return whether a response should be delayed in later stages."""
    return False

