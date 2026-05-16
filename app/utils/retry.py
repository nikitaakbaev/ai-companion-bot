"""Retry utilities."""

from collections.abc import Callable
from typing import TypeVar

from tenacity import retry, stop_after_attempt

T = TypeVar("T")


def retry_three_times(func: Callable[..., T]) -> Callable[..., T]:
    """Retry a synchronous function three times."""
    return retry(stop=stop_after_attempt(3))(func)

