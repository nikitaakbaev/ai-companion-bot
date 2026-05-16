"""Relevance scoring placeholders."""


def is_relevant(score: float, threshold: float) -> bool:
    """Return whether a similarity score passes the configured threshold."""
    return score >= threshold

