"""JSON utilities."""

import orjson


def dumps(data: object) -> str:
    """Serialize data to JSON text."""
    return orjson.dumps(data).decode("utf-8")


def loads(data: str) -> object:
    """Deserialize JSON text."""
    return orjson.loads(data)

